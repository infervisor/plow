//! Code-object naming, packet requirements and compatibility validation.

use std::collections::BTreeSet;
use std::path::Path;

use packet::dev::{DevInst64, DevOp, SE_XCTR};

use super::derive_segments;
use crate::asset::devblob::DevProg;
use crate::{Result, RuntimeError};

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
    pub(super) fn suffix(self) -> &'static str {
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
    pub(super) const fn interpreter_threads(self) -> u32 {
        match self {
            Self::Flash => 4 * 64,
            Self::Prefill | Self::Decode => 8 * 64,
        }
    }

    pub(super) fn symbol_base(self) -> &'static str {
        match self {
            Phase::Prefill => "plow_interp",
            Phase::Decode => "plow_interp_dec",
            Phase::Flash => "plow_interp_flash",
        }
    }

    pub(super) fn object_stem(self) -> &'static str {
        match self {
            Phase::Prefill => "interp_prefill",
            Phase::Decode => "interp_decode",
            Phase::Flash => "interp_flash",
        }
    }
}

pub(super) fn check_interpreter_waves(waves: Option<u32>, phase: Phase, path: &Path) -> Result<()> {
    let expected = phase.interpreter_threads() / 64;
    if waves != Some(expected) {
        return Err(RuntimeError::Device(format!(
            "{}: {phase:?} segment requires {expected} waves, but plow_geom_PLOW_WG_WAVES is {waves:?}; rebuild the matching interpreter object",
            path.display()
        )));
    }
    Ok(())
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
    pub(super) fn infix(self) -> &'static str {
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
}

impl PrefillArm {
    pub(super) fn infix(self) -> &'static str {
        match self {
            PrefillArm::None => "",
            PrefillArm::Mla => "_mla",
            PrefillArm::MlaMoe => "_mla_moe",
            // `_k3` already implies the MLA prefill arms — see `_hs_ax_mla_k3`
            // in runtime/CMakeLists.txt — so it does not stack with `_mla`.
            PrefillArm::K3 => "_k3",
            PrefillArm::K3Moe => "_k3_moe",
            PrefillArm::K3MoeA4w4 => "_k3_moe_a4w4",
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
/// ([`crate::exec::gpu::GpuEngine::check_packet_pairing`]). There is no stamp here —
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
pub(super) const MOE_PF_ENC_SLOT: usize = 3;

/// `PLOW_MOE_ENC_MXFP4` (`runtime/amd/op_moe.h`) — the encoding whose grouped body is compiled
/// only under `PLOW_MOE_PF_A4W4`, i.e. the one that selects a different OBJECT rather than a
/// different branch.
pub(super) const MOE_ENC_MXFP4: u32 = 2;

pub(super) const PREFILL_ARM_MARKERS: &[(&str, &[&str])] = &[
    (
        "PLOW_HAS_NORM_RESIDUAL_NORM",
        &["plow_prefill_nrn_consumer_1"],
    ),
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
    ("PLOW_KDA_CHUNK", &["plow_kda_chunk_bt64_arm_1"]),
    ("PLOW_KDA_CHUNK_QPRE", &["plow_kda_chunk_qpre_arm_1"]),
    // Sequence-parallel seam arms (ops 25/26), compiled only when the packet's plow_config.h
    // carries the ops. Without them the dispatch falls through silently: no reduce, no
    // gather, a prefill of stale slots. Refuse.
    ("PLOW_SEQ_PAR_SEAMS", &["plow_seq_par_seams_arm_1"]),
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
pub(super) const COMPILED_OPCODE_MARKERS: &[(DevOp, &str)] = &[
    (DevOp::KdaConv, "plow_opcode_kda_conv_1"),
    (DevOp::KdaGate, "plow_opcode_kda_gate_1"),
    (DevOp::KdaStateStep, "plow_opcode_kda_state_step_1"),
    (DevOp::KdaGatedNorm, "plow_opcode_kda_gated_norm_1"),
    (DevOp::AttnRes, "plow_opcode_attn_res_1"),
    (DevOp::SituGlu, "plow_opcode_situ_glu_1"),
    (DevOp::MlaOutGate, "plow_opcode_mla_out_gate_1"),
    (DevOp::KdaConv3, "plow_opcode_kda_conv3_1"),
    (DevOp::KdaStateStepG, "plow_opcode_kda_state_step_g_1"),
    (
        DevOp::KdaConvStateStepG,
        "plow_opcode_kda_conv_state_step_g_1",
    ),
    (DevOp::KdaChunkPrepare, "plow_opcode_kda_chunk_prepare_1"),
    (DevOp::KdaChunkIntra, "plow_opcode_kda_chunk_intra_1"),
    (DevOp::KdaChunkWu, "plow_opcode_kda_chunk_wu_1"),
    (DevOp::KdaChunkCarry, "plow_opcode_kda_chunk_carry_1"),
];

pub(super) fn check_compiled_opcode_marker_set(
    syms: &[&str],
    path: &Path,
    required: impl IntoIterator<Item = DevOp>,
) -> Result<()> {
    for op in required {
        let Some((_, marker)) = COMPILED_OPCODE_MARKERS
            .iter()
            .find(|(candidate, _)| *candidate == op)
        else {
            continue;
        };
        if !syms.contains(marker) {
            return Err(RuntimeError::Device(format!(
                "packet/object MISMATCH: packet dispatches {op:?}, but {} lacks compiled-opcode marker `{marker}`",
                path.display()
            )));
        }
    }
    Ok(())
}

pub(super) fn check_compiled_opcode_markers(
    syms: &[&str],
    path: &Path,
    progs: &[DevProg],
) -> Result<()> {
    let required = progs
        .iter()
        .flat_map(|prog| &prog.insts)
        .filter_map(|inst| DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op));
    check_compiled_opcode_marker_set(syms, path, required)
}

pub(super) const MATERIALIZED_RESIDUAL_INPUT_SYM: &str = "plow_materialized_residual_input_1";

pub(super) fn check_materialized_residual_input(
    syms: &[&str],
    path: &Path,
    progs: &[DevProg],
) -> Result<()> {
    let required = progs.iter().flat_map(|prog| &prog.insts).any(|inst| {
        inst.op == DevOp::AttnRes as u16
            && (inst.t[6] != packet::dev::TENSOR_NONE16
                || inst.t[7] != packet::dev::TENSOR_NONE16
                || inst.i[5] != packet::dev::TENSOR_NONE_I)
    });
    if required && !syms.contains(&MATERIALIZED_RESIDUAL_INPUT_SYM) {
        return Err(RuntimeError::Device(format!(
            "packet/object MISMATCH: packet carries a graph-fused materialized residual input, but {} lacks `{MATERIALIZED_RESIDUAL_INPUT_SYM}`",
            path.display()
        )));
    }
    Ok(())
}

/// `op_collective.h` exports this from a `PLOW_XR_TAGGED=1` decode object: its one-shot
/// XReduce arm is the tagged form, which needs `PlowProgram::xr_tag_off` and the blob
/// contract [`crate::exec::amd_tp::check_xr_tagged_blob`] enforces.
pub(super) const XR_TAGGED_SYM: &str = "plow_xr_tagged_1";

pub(super) const DECODE_ARM_MARKERS: &[(&str, &[&str])] = &[
    ("PLOW_DSA_SELECT_LOCAL", &["plow_dsa_select_local_arm"]),
    ("PLOW_KDA_CONV_STEP_DB", &["plow_kda_conv_step_db_arm"]),
    ("PLOW_MOE_PF_ATOMIC", &["plow_moe_pf_atomic_arm"]),
    ("PLOW_MOE_PF_DET", &["plow_moe_pf_det_arm"]),
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
    // The K3 latent MoeCombine folded into the tagged one-shot publish (XReduce t1 = part,
    // i7 = k). An object without the arm publishes the plain partial slot, which no packet
    // wrote — stale data, no trap. A BUILD axis (`#if PLOW_XR_COMBINE_FOLD`).
    ("PLOW_XR_COMBINE_FOLD", &["plow_xr_combine_fold_1"]),
    // The K3 f_b GEMV folded into KdaStateStepG's prologue (flags bit 2, t4 = f_a, j1 = W_fb).
    // An object without the arm reads f_a's 128 values as the head's gate logits — finite,
    // no trap, wrong. A BUILD axis (`#if PLOW_KDA_FB_FOLD`).
    ("PLOW_KDA_FB_FOLD", &["plow_kda_fb_fold_1"]),
    // The K3 Conv3+StepG+GatedNorm chain as one KdaStateStepG packet (flags bit 3, t0 = y,
    // t1..t3 raw q/k/v, t7 = descriptor). An object without the arm runs the recurrence on the
    // raw projections with the descriptor as A_log — finite, no trap, wrong. A BUILD axis
    // (`#if PLOW_KDA_DECODE_FUSED_ARM`).
    (
        "PLOW_KDA_DECODE_FUSED_ARM",
        &["plow_kda_decode_fused_arm_1"],
    ),
    // The decode fused QKV/GLU emitted against a STATED staging arena. The marker carries the
    // arena as a VALUE (`check_dec_stage_capacity` compares M*K against it); this entry is the
    // other half — an object old enough not to publish the arena cannot be checked at all, and
    // the fused bodies stage x past the end of LDS instead of trapping.
    ("PLOW_DEC_STAGE_HALVES", &[DEC_STAGE_SYM]),
];

pub(super) fn required_moe_pf_accum(progs: &[DevProg], field: usize) -> bool {
    progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(|inst| inst.op == DevOp::MoeGroupDownPf as u16 && inst.i[field] != 0)
}

/// Refuse a DECODE code object that does not carry the arms the packet needs.
///
/// See [`check_prefill_object`] for the full argument — this is that check, on the other object,
/// and it ignores any flag [`DECODE_ARM_MARKERS`] does not name (those belong to the prefill
/// object, which gets its own pass).
pub(super) fn check_decode_object(
    syms: &[&str],
    path: &Path,
    requires: &[String],
    need_moe_pf_atomic: bool,
    need_moe_pf_det: bool,
) -> Result<()> {
    if syms.is_empty() {
        let needs_decode_arm = requires.iter().any(|req| {
            let (flag, val) = req.split_once('=').unwrap_or((req.as_str(), "1"));
            val != "0"
                && DECODE_ARM_MARKERS
                    .iter()
                    .any(|(candidate, _)| *candidate == flag)
        });
        if needs_decode_arm {
            return Err(RuntimeError::Device(format!(
                "packet requires a specialised decode arm, but {} has no ELF symbol table; refusing an unverifiable packet/object pairing",
                path.display()
            )));
        }
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
        // `requires` is blob-wide; a B1 decode object must not inherit an arm used only
        // by grouped prefill programs.
        if (flag == "PLOW_MOE_PF_ATOMIC" && !need_moe_pf_atomic)
            || (flag == "PLOW_MOE_PF_DET" && !need_moe_pf_det)
        {
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
pub(super) fn manifest_pairing_hash(raw: &[u8], path: &Path) -> Result<u64> {
    let manifest: serde_json::Value = serde_json::from_slice(raw)
        .map_err(|e| RuntimeError::Device(format!("{}: not valid JSON: {e}", path.display())))?;
    manifest
        .pointer("/pairing/hash")
        .and_then(|v| v.as_str())
        .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
        .ok_or_else(|| {
            RuntimeError::Device(format!(
                "specialised AMD object requires {} to contain a valid pairing.hash",
                path.display()
            ))
        })
}

pub(super) fn validate_packet_pairing_stamp(
    lo: Option<u32>,
    hi: Option<u32>,
    expected: Option<u64>,
    object: &Path,
) -> Result<()> {
    match (lo, hi) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => Err(RuntimeError::Device(format!(
            "{} has a partial packet-pairing stamp; refusing an unverifiable specialised object",
            object.display()
        ))),
        (Some(lo), Some(hi)) => {
            let stamped = ((hi as u64) << 32) | lo as u64;
            let expected = expected.ok_or_else(|| {
                RuntimeError::Device(format!(
                    "{} is specialised for packet hash 0x{stamped:016x}, but the asset has no valid build.json pairing hash",
                    object.display()
                ))
            })?;
            if stamped != expected {
                return Err(RuntimeError::Device(format!(
                    "packet/object MISMATCH: specialised AMD object {} stamps 0x{stamped:016x}, asset requires 0x{expected:016x}",
                    object.display()
                )));
            }
            Ok(())
        }
    }
}

pub(super) fn build_pairing_hash(blob_path: &Path) -> Result<Option<u64>> {
    let path = blob_path.with_file_name("build.json");
    match std::fs::read(&path) {
        Ok(raw) => manifest_pairing_hash(&raw, &path).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(RuntimeError::Io { path, source }),
    }
}

pub(super) fn check_packet_pairing_stamp(
    image: &[u8],
    blob_path: &Path,
    object: &Path,
) -> Result<()> {
    let lo = elf_symbol_u32(image, "plow_packet_hash_lo");
    let hi = elf_symbol_u32(image, "plow_packet_hash_hi");
    let expected = if lo.is_some() || hi.is_some() {
        build_pairing_hash(blob_path)?
    } else {
        None
    };
    validate_packet_pairing_stamp(lo, hi, expected, object)
}

pub(super) const KDA_INTRA_WAVE_ITEMS_MARKERS: [&str; 6] = [
    "plow_kda_intra_wave_items_abi_1",
    "plow_kda_intra_wave_items_bt64_d128_1",
    "plow_kda_intra_wave_items_wave64_1",
    "plow_kda_intra_wave_items_no_spill_1",
    "plow_kda_intra_wave_items_static_lds_131072",
    "plow_kda_intra_wave_items_vgpr_le_96",
];

pub(super) const KDA_CARRY_REGSTATE_MARKERS: [&str; 6] = [
    "plow_kda_carry_regstate_abi_1",
    "plow_kda_carry_regstate_bt64_d128_v128_qpre_1",
    "plow_kda_carry_regstate_wave64_1",
    "plow_kda_carry_regstate_no_spill_1",
    "plow_kda_carry_regstate_static_lds_43520",
    "plow_kda_carry_regstate_vgpr_le_256",
];
pub(super) const KDA_CARRY_REGSTATE_LDS: u32 = 43_520;

pub(super) const KDA_WU_LEAN_MARKERS: [&str; 6] = [
    "plow_kda_wu_lean_abi_1",
    "plow_kda_wu_lean_bt64_d128_v128_qpre_1",
    "plow_kda_wu_lean_wave64_1",
    "plow_kda_wu_lean_no_spill_1",
    "plow_kda_wu_lean_static_lds_46080",
    "plow_kda_wu_lean_vgpr_le_256",
];
pub(super) const KDA_WU_LEAN_LDS: u32 = 46_080;

pub(super) const KDA_CARRY_KEYFEED_MARKERS: [&str; 7] = [
    "plow_kda_carry_keyfeed_abi_1",
    "plow_kda_carry_keyfeed_bt64_d128_v128_qpre_1",
    "plow_kda_carry_keyfeed_wave64_1",
    "plow_kda_carry_keyfeed_no_spill_1",
    "plow_kda_carry_keyfeed_static_lds_43520",
    "plow_kda_carry_keyfeed_vgpr_le_256",
    "plow_kda_carry_keyfeed_scratch_pair_bf16_1",
];

pub(super) fn read_kda_carry_regstate_object(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| {
        RuntimeError::Device(format!(
            "marked KDA carry regstate segments require {}: {e}",
            path.display()
        ))
    })
}

pub(super) fn check_kda_carry_regstate_symbols<'a>(syms: &[&'a str], path: &Path) -> Result<()> {
    for marker in KDA_CARRY_REGSTATE_MARKERS {
        if !syms.contains(&marker) {
            return Err(RuntimeError::Device(format!(
                "KDA carry regstate object {} lacks required ABI/resource marker `{marker}`",
                path.display()
            )));
        }
    }
    if !syms.contains(&"plow_packet_hash_lo") || !syms.contains(&"plow_packet_hash_hi") {
        return Err(RuntimeError::Device(format!(
            "KDA carry regstate object {} has no packet-pairing stamp",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn read_kda_intra_wave_items_object(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| {
        RuntimeError::Device(format!(
            "marked KDA-intra wave-item segments require {}: {e}",
            path.display()
        ))
    })
}

pub(super) fn check_kda_intra_wave_items_symbols<'a>(syms: &[&'a str], path: &Path) -> Result<()> {
    for marker in KDA_INTRA_WAVE_ITEMS_MARKERS {
        if !syms.contains(&marker) {
            return Err(RuntimeError::Device(format!(
                "KDA-intra wave-item object {} lacks required ABI/resource marker `{marker}`",
                path.display()
            )));
        }
    }
    if !syms.contains(&"plow_packet_hash_lo") || !syms.contains(&"plow_packet_hash_hi") {
        return Err(RuntimeError::Device(format!(
            "KDA-intra wave-item object {} has no packet-pairing stamp",
            path.display()
        )));
    }
    Ok(())
}

pub(super) const ATTN_RES_F32MIX_MARKERS: [&str; 6] = [
    "plow_attn_res_f32mix_abi_1",
    "plow_attn_res_f32mix_hid7168_1",
    "plow_attn_res_f32mix_online_softmax_1",
    "plow_attn_res_f32mix_wave64_1",
    "plow_attn_res_f32mix_no_spill_1",
    "plow_attn_res_f32mix_vgpr_le_168",
];

pub(super) fn read_attn_res_f32mix_object(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| {
        RuntimeError::Device(format!(
            "f32-mix AttnRes packets require {}: {e}",
            path.display()
        ))
    })
}

pub(super) fn check_attn_res_f32mix_symbols(syms: &[&str], path: &Path) -> Result<()> {
    for marker in ATTN_RES_F32MIX_MARKERS {
        if !syms.contains(&marker) {
            return Err(RuntimeError::Device(format!(
                "f32-mix AttnRes object {} lacks required ABI/resource marker `{marker}`",
                path.display()
            )));
        }
    }
    if !syms.contains(&"plow_packet_hash_lo") || !syms.contains(&"plow_packet_hash_hi") {
        return Err(RuntimeError::Device(format!(
            "f32-mix AttnRes object {} has no packet-pairing stamp",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn check_moe_ep_symbols(syms: &[&str], path: &Path, markers: &[&str]) -> Result<()> {
    for marker in markers {
        if !syms.contains(marker) {
            return Err(RuntimeError::Device(format!(
                "EP object {} lacks required ABI/resource marker {marker}",
                path.display()
            )));
        }
    }
    Ok(())
}

pub(super) fn build_requires(blob_path: &Path) -> Result<Option<Vec<String>>> {
    let mpath = blob_path.with_file_name("build.json");
    let raw = match std::fs::read(&mpath) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(RuntimeError::Device(format!(
                "cannot read {}: {e}",
                mpath.display()
            )))
        }
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

pub(super) fn packet_decode_arm_requirements(progs: &[DevProg]) -> Vec<String> {
    let insts = || progs.iter().flat_map(|p| &p.insts);
    let mut requires = Vec::new();
    if insts().any(|inst| inst.op == DevOp::IndexSelect as u16 && inst.i[4] == 1) {
        requires.push("PLOW_DSA_SELECT_LOCAL=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::KdaConvStateStepG as u16) {
        requires.push("PLOW_KDA_CONV_STEP_DB=1".to_owned());
    }
    if required_moe_pf_accum(progs, 4) {
        requires.push("PLOW_MOE_PF_ATOMIC=1".to_owned());
    }
    if required_moe_pf_accum(progs, 5) {
        requires.push("PLOW_MOE_PF_DET=1".to_owned());
    }
    if insts().any(|inst| {
        inst.op == DevOp::FlashMlaDecode as u16 && inst.t[7] != packet::dev::TENSOR_NONE16
    }) {
        requires.push("PLOW_GLM_FUSE_ROPE=1".to_owned());
    }
    if insts()
        .any(|inst| inst.op == DevOp::GemvQkv as u16 && inst.t[7] != packet::dev::TENSOR_NONE16)
    {
        requires.push("PLOW_GLM_FUSE_QNORM=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::XReduce as u16 && inst.i[7] != 0) {
        requires.push("PLOW_XR_COMBINE_FOLD=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::KdaStateStepG as u16 && inst.i[4] & 4 != 0) {
        requires.push("PLOW_KDA_FB_FOLD=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::KdaStateStepG as u16 && inst.i[4] & 8 != 0) {
        requires.push("PLOW_KDA_DECODE_FUSED_ARM=1".to_owned());
    }
    requires
}

pub(super) fn packet_prefill_arm_requirements(progs: &[DevProg]) -> Vec<String> {
    let insts = || progs.iter().flat_map(|p| &p.insts);
    let mut requires = Vec::new();
    if insts().any(|inst| inst.op == DevOp::NormResidualNorm as u16) {
        requires.push("PLOW_HAS_NORM_RESIDUAL_NORM=1".to_owned());
    }
    if insts().any(|inst| {
        matches!(
            DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op),
            Some(DevOp::FlashMlaPrefill | DevOp::FlashMlaPrefillFp8 | DevOp::FlashGatherPrefill)
        )
    }) {
        requires.push("PLOW_MLA_PREFILL=1".to_owned());
    }
    if insts().any(|inst| {
        matches!(
            DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op),
            Some(
                DevOp::MoeRouterTopkPf
                    | DevOp::MoeAlignPf
                    | DevOp::MoeGroupGluPf
                    | DevOp::MoeGroupDownPf
                    | DevOp::MoeCombinePf
            )
        )
    }) {
        requires.push("PLOW_MOE_PREFILL=1".to_owned());
    }
    if insts().any(|inst| {
        matches!(inst.op, op if op == DevOp::MoeGroupDownPf as u16 || op == DevOp::MoeCombinePf as u16)
            && inst.i[7] != 0
    }) {
        requires.push("PLOW_MOE_PF_PART16=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::MoeGroupGluPf as u16 && inst.i[7] != 0) {
        requires.push("PLOW_MOE_PF_A8=1".to_owned());
    }
    if insts().any(|inst| {
        inst.op == DevOp::FlashMlaPrefill as u16
            && inst.t[7] == packet::dev::TENSOR_NONE16
            && (inst.i[6] & 0xff) > 1
    }) {
        requires.push("PLOW_MLA_PF_NS=1".to_owned());
    }
    if insts().any(|inst| inst.op == DevOp::FlashMlaPrefillFp8 as u16 && inst.fj[2] != 0) {
        requires.push("PLOW_MLA_PREFILL_FP8_SPLIT=1".to_owned());
    }
    if insts()
        .any(|inst| inst.op == DevOp::QuantFp8 as u16 && inst.t[3] != packet::dev::TENSOR_NONE16)
    {
        requires.push("PLOW_T11_GLUQUANT=1".to_owned());
    }
    if required_moe_pf_accum(progs, 4) {
        requires.push("PLOW_MOE_PF_ATOMIC=1".to_owned());
    }
    if required_moe_pf_accum(progs, 5) {
        requires.push("PLOW_MOE_PF_DET=1".to_owned());
    }
    if insts().any(|inst| {
        matches!(
            DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op),
            Some(
                DevOp::KdaChunkPrepare
                    | DevOp::KdaChunkIntra
                    | DevOp::KdaChunkWu
                    | DevOp::KdaChunkCarry
            )
        )
    }) {
        requires.push("PLOW_KDA_CHUNK=1".to_owned());
    }
    if insts()
        .any(|inst| inst.op == DevOp::XReduceScatter as u16 || inst.op == DevOp::XAllGather as u16)
    {
        requires.push("PLOW_SEQ_PAR_SEAMS=1".to_owned());
    }
    if insts().any(|inst| {
        matches!(
            DevOp::ALL.iter().copied().find(|op| *op as u16 == inst.op),
            Some(
                DevOp::AttnRes
                    | DevOp::SituGlu
                    | DevOp::MlaOutGate
                    | DevOp::KdaStateStep
                    | DevOp::KdaConv
            )
        )
    }) {
        requires.push("PLOW_K3=1".to_owned());
    }
    requires
}

pub(super) fn graph_phase_xreduce_segments(
    blob_path: &Path,
    progs: &[DevProg],
    dec_ix: usize,
    enabled: bool,
) -> Result<Vec<BTreeSet<usize>>> {
    let selected = vec![BTreeSet::new(); progs.len()];
    if !enabled {
        return Ok(selected);
    }
    let mpath = blob_path.with_file_name("build.json");
    let raw = std::fs::read(&mpath).map_err(|source| RuntimeError::Io {
        path: mpath.clone(),
        source,
    })?;
    let man: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| RuntimeError::Device(format!("{}: not valid JSON: {e}", mpath.display())))?;
    graph_phase_xreduce_segments_from_manifest(&man, progs, dec_ix, &mpath)
}

pub(super) fn graph_phase_xreduce_segments_from_manifest(
    man: &serde_json::Value,
    progs: &[DevProg],
    dec_ix: usize,
    mpath: &Path,
) -> Result<Vec<BTreeSet<usize>>> {
    let mut selected = vec![BTreeSet::new(); progs.len()];
    let chains = man
        .get("dispatch_chains")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            RuntimeError::Device(format!(
                "{} has no compiler-derived dispatch_chains; refusing phase-object selection",
                mpath.display()
            ))
        })?;
    if chains.len() != progs.len() {
        return Err(RuntimeError::Device(format!(
            "{} has {} dispatch chains for {} packet programs",
            mpath.display(),
            chains.len(),
            progs.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for chain in chains {
        let program = chain
            .get("program")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RuntimeError::Device("dispatch chain has no program index".into()))?
            as usize;
        if program >= progs.len() || !seen.insert(program) {
            return Err(RuntimeError::Device(format!(
                "invalid or duplicate dispatch-chain program {program}"
            )));
        }
        let expected_kind = if program < dec_ix {
            "prefill"
        } else {
            "decode"
        };
        let expected_topology = if progs[program].packed_prefill_only {
            "packed"
        } else {
            "ordinary"
        };
        if chain.get("kind").and_then(|v| v.as_str()) != Some(expected_kind)
            || chain.get("topology").and_then(|v| v.as_str()) != Some(expected_topology)
        {
            return Err(RuntimeError::Device(format!(
                "dispatch-chain topology mismatch for program {program}"
            )));
        }
        let n_segments = derive_segments(&progs[program])?.len();
        let segment_rows = chain
            .get("segments")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RuntimeError::Device(format!("program {program} has no segments")))?;
        if segment_rows.len() != n_segments {
            return Err(RuntimeError::Device(format!(
                "program {program} manifest has {} segments, packet has {n_segments}",
                segment_rows.len()
            )));
        }
        for (seg, row) in segment_rows.iter().enumerate() {
            if row.get("segment").and_then(|v| v.as_u64()) != Some(seg as u64) {
                return Err(RuntimeError::Device(format!(
                    "program {program} dispatch segments are not contiguous at {seg}"
                )));
            }
        }
        if program >= dec_ix || progs[program].packed_prefill_only {
            continue;
        }
        let phases = chain
            .get("phases")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RuntimeError::Device(format!("program {program} has no phases")))?;
        let mut next_phase_segment = 0usize;
        for phase in phases {
            let lo = phase
                .get("first_segment")
                .and_then(|v| v.as_u64())
                .unwrap_or(u64::MAX) as usize;
            let hi = phase
                .get("last_segment")
                .and_then(|v| v.as_u64())
                .unwrap_or(u64::MAX) as usize;
            let phase_segments = phase
                .get("segments")
                .and_then(|v| v.as_u64())
                .unwrap_or(u64::MAX) as usize;
            if lo != next_phase_segment
                || lo > hi
                || hi >= n_segments
                || phase_segments != hi - lo + 1
            {
                return Err(RuntimeError::Device(format!(
                    "program {program} phases do not form a contiguous partition at {next_phase_segment}"
                )));
            }
            next_phase_segment = hi + 1;
            let families = phase.get("families").and_then(|v| v.as_array());
            let arms = phase.get("arms").and_then(|v| v.as_array());
            let collective_only = families
                .is_some_and(|f| f.len() == 1 && f[0].as_str() == Some("collective"))
                && arms.is_some_and(|a| {
                    !a.is_empty()
                        && a.iter().all(|v| {
                            v.as_str()
                                .is_some_and(|s| s.split('/').next() == Some("XReduceTwoShot"))
                        })
                });
            if !collective_only {
                continue;
            }
            if phase.get("object_class").and_then(|v| v.as_str()) != Some("ordinary") {
                return Err(RuntimeError::Device(format!(
                    "program {program} collective phase is not an ordinary object class"
                )));
            }
            let contract = phase.get("resource_contract").ok_or_else(|| {
                RuntimeError::Device(format!(
                    "program {program} collective phase has no contract"
                ))
            })?;
            let contract_ok = contract.get("policy").and_then(|v| v.as_str()) == Some("refuse")
                && contract.get("wavefront_size").and_then(|v| v.as_u64()) == Some(64)
                && contract
                    .get("min_occupancy_waves_per_simd")
                    .and_then(|v| v.as_u64())
                    .is_some_and(|v| v >= 2)
                && contract
                    .get("max_private_segment_bytes_delta")
                    .and_then(|v| v.as_u64())
                    == Some(0)
                && contract
                    .get("max_vgpr_spill_delta")
                    .and_then(|v| v.as_u64())
                    == Some(0)
                && contract
                    .get("max_sgpr_spill_delta")
                    .and_then(|v| v.as_u64())
                    == Some(0);
            if !contract_ok {
                return Err(RuntimeError::Device(format!(
                    "program {program} collective phase has an incompatible resource contract"
                )));
            }
            for seg in lo..=hi {
                let row = &segment_rows[seg];
                let row_families = row.get("families").and_then(|v| v.as_array());
                let row_arms = row.get("arms").and_then(|v| v.as_array());
                if !row_families
                    .is_some_and(|f| f.len() == 1 && f[0].as_str() == Some("collective"))
                    || !row_arms.is_some_and(|a| {
                        !a.is_empty()
                            && a.iter().all(|v| {
                                v.as_str()
                                    .is_some_and(|s| s.split('/').next() == Some("XReduceTwoShot"))
                            })
                    })
                {
                    return Err(RuntimeError::Device(format!(
                        "program {program} phase claims collective-only segment {seg}, but its segment inventory differs"
                    )));
                }
                let mut entries = progs[program]
                    .stream
                    .iter()
                    .filter(|e| e.seg as usize == seg)
                    .peekable();
                if entries.peek().is_none()
                    || entries.any(|e| {
                        progs[program]
                            .insts
                            .get(e.inst as usize)
                            .is_none_or(|inst| inst.op != DevOp::XReduceTwoShot as u16)
                    })
                {
                    return Err(RuntimeError::Device(format!(
                        "program {program} packet segment {seg} is not XReduceTwoShot-only"
                    )));
                }
            }
            selected[program].extend(lo..=hi);
        }
        if next_phase_segment != n_segments {
            return Err(RuntimeError::Device(format!(
                "program {program} phase partition covers {next_phase_segment} of {n_segments} segments"
            )));
        }
    }
    Ok(selected)
}

/// Every symbol NAME in an ELF64 object's symbol tables.
///
/// A deliberately minimal reader rather than a dependency: it needs the section
/// headers, the two symbol-table types and their string tables, and every bound
/// is checked so a truncated or foreign file yields an empty list instead of a
/// panic. A file this cannot parse is reported by the CALLER as "unverifiable",
/// never as "the arm is missing" — a parser bug must not become a refusal.
pub(in crate::exec) fn elf_symbol_names(img: &[u8]) -> Vec<&str> {
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

/// Initial value of a four-byte ELF object symbol.
///
/// Pairing stamps are immutable build metadata. Reading them from the file avoids loading a
/// module and asking SDMA to read a device global merely to compare two constants.
pub(super) fn elf_symbol_u32(img: &[u8], wanted: &str) -> Option<u32> {
    let u16at = |o: usize| -> Option<usize> {
        img.get(o..o + 2)
            .and_then(|b| Some(u16::from_le_bytes(b.try_into().ok()?) as usize))
    };
    let u32at = |o: usize| -> Option<usize> {
        img.get(o..o + 4)
            .and_then(|b| Some(u32::from_le_bytes(b.try_into().ok()?) as usize))
    };
    let u64at = |o: usize| -> Option<usize> {
        img.get(o..o + 8)
            .and_then(|b| usize::try_from(u64::from_le_bytes(b.try_into().ok()?)).ok())
    };
    if img.get(..4) != Some(b"\x7fELF") || img.get(4) != Some(&2) || img.get(5) != Some(&1) {
        return None;
    }
    let (shoff, shent, shnum) = (u64at(0x28)?, u16at(0x3a)?, u16at(0x3c)?);
    if shent < 64 {
        return None;
    }
    let hdr = |i: usize| -> Option<usize> {
        (i < shnum)
            .then(|| i.checked_mul(shent)?.checked_add(shoff))
            .flatten()
            .filter(|&off| off.checked_add(64).is_some_and(|end| end <= img.len()))
    };
    for i in 0..shnum {
        let s = hdr(i)?;
        if !matches!(u32at(s + 4), Some(2) | Some(11)) {
            continue;
        }
        let (off, size, link, entsz) = (
            u64at(s + 24)?,
            u64at(s + 32)?,
            u32at(s + 40)?,
            u64at(s + 56)?,
        );
        if entsz < 24 {
            continue;
        }
        let l = hdr(link)?;
        let (stroff, strsz) = (u64at(l + 24)?, u64at(l + 32)?);
        let strtab = img.get(stroff..stroff.checked_add(strsz)?)?;
        for k in 0..size / entsz {
            let sym = off.checked_add(k.checked_mul(entsz)?)?;
            let nm = u32at(sym)?;
            let tail = strtab.get(nm..)?;
            let end = tail.iter().position(|&c| c == 0).unwrap_or(tail.len());
            if tail.get(..end)? != wanted.as_bytes() {
                continue;
            }
            if u64at(sym + 16)? != 4 {
                return None;
            }
            let section = hdr(u16at(sym + 6)?)?;
            if u32at(section + 4)? != 1 {
                return None;
            }
            let (section_addr, section_off, section_size) = (
                u64at(section + 16)?,
                u64at(section + 24)?,
                u64at(section + 32)?,
            );
            let rel = u64at(sym + 8)?.checked_sub(section_addr)?;
            if rel.checked_add(4)? > section_size {
                return None;
            }
            let value_off = section_off.checked_add(rel)?;
            return img
                .get(value_off..value_off.checked_add(4)?)
                .map(|b| u32::from_le_bytes(b.try_into().expect("four-byte symbol")));
        }
    }
    None
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
pub(super) fn check_prefill_object(syms: &[&str], path: &Path, requires: &[String]) -> Result<()> {
    if syms.is_empty() {
        let needs_prefill_arm = requires.iter().any(|req| {
            let (flag, val) = req.split_once('=').unwrap_or((req.as_str(), "1"));
            val != "0"
                && PREFILL_ARM_MARKERS
                    .iter()
                    .any(|(candidate, _)| *candidate == flag)
        });
        if needs_prefill_arm {
            return Err(RuntimeError::Device(format!(
                "packet requires a specialised prefill arm, but {} has no ELF symbol table; refusing an unverifiable packet/object pairing",
                path.display()
            )));
        }
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
        if flag == "PLOW_MLA_PREFILL_FP8_SPLIT" {
            // Validated against the dedicated split object by segment::load_small_mla.
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
            "These flags are outside the opcode-arm check; phase selection and interpreter \
             wave geometry are validated separately."
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
pub(super) const GEMV_CAP_SYM_PREFIX: &str = "plow_gemv_mm_cap_";

/// The `extern "C" __device__` marker `runtime/amd/op_gemm.h` emits when it was
/// compiled with `PLOW_GEMV_WALK=1` — the single-rung outer loop over `M > MM`.
///
/// Present iff the macro is on, so absence is not ambiguous: every object built
/// before the walk existed, and every object built with it off, carries no such
/// symbol and is a hard-capacity object exactly as before.
pub(super) const GEMV_WALK_SYM: &str = "plow_gemv_walk_1";
pub(super) const XARGMAX_B128_SYM: &str = "plow_xargmax_max_batch_128";
pub(super) const PACKED_PREFILL_ABI_SYM: &str = "plow_packed_prefill_abi_1";
pub(super) const PACKED_PREFILL_MLA_NORM_SEG_SYM: &str = "plow_packed_prefill_mla_norm_segments_1";
/// The slot-band row resolver (`PLOW_PACKED_PREFILL_BAND=1`), carried by the `_tb` family
/// objects a token-batch body program is routed to.
pub(super) const PACKED_PREFILL_BAND_SYM: &str = "plow_packed_prefill_band_1";
pub(super) const PACKED_PREFILL_MLA_FLASH_SEG_SYM: &str =
    "plow_packed_prefill_mla_flash_segments_1";
pub(super) const PACKED_PREFILL_KDA_SEG_SYM: &str = "plow_packed_prefill_kda_serial_segments_1";
pub(super) const PACKED_PREFILL_KDA_CHUNK_SEG_SYM: &str =
    "plow_packed_prefill_kda_chunk_segments_1";
pub(super) const KDA_FAMILY_SEG_SYM: &str = "plow_kda_family_segments_1";
pub(super) const XR_ATTNRES_SEG_SYM: &str = "plow_xr_attnres_segment_1";
pub(super) const XR_ATTNRES_RESOURCE_SYM: &str = "plow_xr_attnres_wave64_nospill_1";

pub(super) fn check_packed_prefill_abi(
    prefill: bool,
    flash_required: bool,
    flash: bool,
) -> Result<()> {
    if !prefill {
        return Err(RuntimeError::Device(format!(
            "packed-prefill metadata requires a prefill object advertising \
             `{PACKED_PREFILL_ABI_SYM}`; rebuild the AMD objects before staging it"
        )));
    }
    if flash_required && !flash {
        return Err(RuntimeError::Device(format!(
            "packed-prefill program routes a segment to a flash object that does not advertise \
             `{PACKED_PREFILL_ABI_SYM}`; rebuild every routed AMD object before staging it"
        )));
    }
    Ok(())
}

pub(super) fn check_packed_family_kv_encoding(has_fp8: bool, want_fp8: bool) -> Result<()> {
    if has_fp8 == want_fp8 {
        Ok(())
    } else {
        Err(RuntimeError::Device(format!(
            "packed-prefill family object has fp8-KV capability {has_fp8}, packet requires \
             {want_fp8}; refusing an encoding mismatch"
        )))
    }
}

/// `PLOW_GEMV_MAXM` from `runtime/amd/op_gemm.h` — the widest bucket the GEMV
/// path can be instantiated at, and therefore the widest any object can
/// advertise. `scripts/build_gfx950.sh` and `runtime/CMakeLists.txt` both clamp
/// `PLOW_GEMV_MM` to it to satisfy the header's static assert.
///
/// A ceiling, not a pairing: see the use in [`super::AmdEngine::load`], and
/// [`check_gemv_capacity`] for the half that compares against the object.
pub(super) const GEMV_MAXM: u32 = 16;

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
pub(super) const GEMV_BUCKET_OPS: &[DevOp] = &[
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
pub(super) fn required_gemv_m(progs: &[DevProg]) -> u32 {
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
/// The `extern "C" __device__` marker `runtime/amd/interp.hip` emits when it was compiled with
/// `PLOW_K3=1` — the seven Kimi-K3 / KDA arms.
///
/// Present iff the axis is on, so absence is not ambiguous: every object built before the axis
/// existed, and every object built with it off, carries no such symbol and has no K3 arm.
pub(super) const K3_ARMS_SYM: &str = "plow_k3_arms_1";
/// `-DPLOW_QWEN_GDN=1` — Qwen3.5's Gated DeltaNet arms (runtime/amd/op_qwen_gdn.h).
pub(super) const QWEN_GDN_ARMS_SYM: &str = "plow_qwen_gdn_arms_1";
pub(super) const MOE_PF_A4W4_SYM: &str = "plow_moe_pf_a4w4_arm";
pub(super) const KDA_CONV_STEP_DB_SYM: &str = "plow_kda_conv_step_db_arm";
pub(super) const KDA_CHUNK_SYM: &str = "plow_kda_chunk_bt64_arm_1";
pub(super) const KDA_CHUNK_QPRE_SYM: &str = "plow_kda_chunk_qpre_arm_1";
pub(super) const KDA_CONV_STEP_DB_REPLACED_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaConv3,
    DevOp::KdaStateStepG,
];

/// Every opcode that reaches an arm behind `PLOW_K3` in `runtime/amd/interp.hip`.
///
/// The KDA mixer ops — four decomposed plus the two FUSED ones the decode emitter actually uses —
/// the three K3 block-structure ops, and (added alongside GLM-5.3-Flash, which shares this same
/// guard region rather than its own) the eight GLM-5.3 primitives: hyper-connections and the
/// pooled DSA indexer. This is the Rust half of the contract whose C half is the `#if PLOW_K3`
/// region around those `case` labels; `k3_arm_ops_match_the_interpreter` reads `interp.hip` and
/// asserts the two agree, so a future arm added inside the guard cannot go unlisted here — which
/// is exactly how `KdaConv3` and `KdaStateStepG` were forced onto this list rather than
/// remembered onto it, and how the eight GLM-5.3 ops below were found missing: they NOPped
/// silently on any object built without `PLOW_K3` with no load-time refusal, because this list
/// was never updated when they were added to the interpreter — caught only by running this
/// test under `--features hsa`, which this project's own documented verification recipes
/// (`status.md`, `perf-data/vllm-k3-glm53-baseline.md`) never pass.
pub(super) const K3_ARM_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaGatedNorm,
    DevOp::KdaConv3,
    DevOp::KdaStateStepG,
    DevOp::KdaConvStateStepG,
    DevOp::KdaChunkPrepare,
    DevOp::KdaChunkIntra,
    DevOp::KdaChunkWu,
    DevOp::KdaChunkCarry,
    DevOp::AttnRes,
    DevOp::SituGlu,
    DevOp::MlaOutGate,
    DevOp::HyperConnPre,
    DevOp::HyperConnPost,
    DevOp::DsaPoolCompress,
    DevOp::DsaPoolExpand,
    DevOp::DsaPoolStash,
    DevOp::DsaQQuant,
    DevOp::IndexScoreKpool,
    DevOp::GemvF32,
];

/// Every opcode `PLOW_QWEN_GDN` compiles an arm for. `PLOW_DOP_QWEN_GDN_PREFILL` is deliberately
/// absent: it is not an interpreter arm on any backend, so no object symbol can answer for it and
/// the emit refuses it instead (`refuse_unimplemented_target`, crates/devgen/src/qwen35.rs).
pub(super) const QWEN_GDN_ARM_OPS: &[DevOp] = &[
    DevOp::QwenGdnConv,
    DevOp::QwenGdnStep,
    DevOp::QwenGatedNorm,
    DevOp::QwenQGateSplit,
    DevOp::QwenSigmoidGate,
    DevOp::QwenRmsNorm,
    DevOp::QwenHeadNormRope,
    DevOp::QwenGdnConvPrefill,
    DevOp::QwenGdnQkvPrep,
    DevOp::QwenGdnGatePrep,
];

/// The first Qwen GDN opcode in these programs, or `None` if the packet needs no Qwen arm.
pub(super) fn required_qwen_gdn_op(progs: &[DevProg]) -> Option<DevOp> {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find_map(|i| QWEN_GDN_ARM_OPS.iter().copied().find(|&o| o as u16 == i.op))
}

/// The first K3/KDA opcode in these programs, or `None` if the packet needs no K3 arm.
pub(super) fn required_k3_op(progs: &[DevProg]) -> Option<DevOp> {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find_map(|i| K3_ARM_OPS.iter().copied().find(|&o| o as u16 == i.op))
}

pub(super) fn required_moe_pf_a4w4(progs: &[DevProg]) -> Option<DevOp> {
    progs.iter().flat_map(|p| p.insts.iter()).find_map(|i| {
        let op = DevOp::from_u16(i.op)?;
        ((op == DevOp::MoeGroupGluPf || op == DevOp::MoeGroupDownPf)
            && i.i[MOE_PF_ENC_SLOT] == MOE_ENC_MXFP4)
            .then_some(op)
    })
}

pub(super) fn required_kda_conv_step_db(progs: &[DevProg]) -> Option<DevOp> {
    first_op_in(progs, &[DevOp::KdaConvStateStepG])
}

pub(super) fn required_kda_chunk(progs: &[DevProg]) -> Option<DevOp> {
    first_op_in(
        progs,
        &[
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ],
    )
}

pub(super) fn check_kda_chunk(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
    let Some(op) = need else {
        return Ok(());
    };
    if syms.contains(&KDA_CHUNK_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object chunk-KDA MISMATCH: this packet dispatches {op:?} (op {}), but {} does \
         not advertise `{KDA_CHUNK_SYM}`. Rebuild the KDA prefill object with \
         -DPLOW_KDA_CHUNK=1, or emit the serial recurrence.",
        op as u16,
        path.display()
    )))
}

pub(super) fn check_kda_conv_step_db(
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

pub(super) fn check_moe_pf_a4w4(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
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
/// Refuse a Qwen3.5 Gated DeltaNet packet against an object built without `PLOW_QWEN_GDN`.
///
/// Exactly [`check_k3_arms`]'s argument, for exactly the same reason: the family is compiled OUT by
/// default (the arms cost register pressure on every object that carries them, and nothing routes
/// to them yet), and AMD's dispatch `default:` writes NOTHING rather than trapping. Without this,
/// a Qwen packet on an unarmed object would leave every GDN output untouched and the run would
/// complete on uninitialised memory — fluent, and wrong.
pub(super) fn check_qwen_gdn_arms(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
    let Some(op) = need else {
        return Ok(());
    };
    if syms.contains(&QWEN_GDN_ARMS_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object QWEN-GDN MISMATCH: this packet dispatches {op:?} (op {}), but {} was \
         compiled without PLOW_QWEN_GDN (it does not advertise `{QWEN_GDN_ARMS_SYM}`). AMD's \
         dispatch default writes NOTHING rather than trapping, so this op would silently leave its \
         output untouched and the run would complete on uninitialised memory instead of failing. \
         Rebuild the object with -DPLOW_QWEN_GDN=1.",
        op as u16,
        path.display()
    )))
}

pub(super) fn check_k3_arms(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
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

pub(super) fn check_dsa_decode_batch(
    syms: &[&str],
    path: &Path,
    progs: &[DevProg],
    cdna3: bool,
) -> Result<()> {
    if progs.iter().flat_map(|p| &p.insts).any(|d| {
        (d.op == DevOp::IndexSelect as u16 && (d.i[3] != 0 || d.i[4] != 0))
            || (cdna3 && d.op == DevOp::IndexScore as u16)
    }) && !syms.contains(&"plow_dsa_decode_batch_arm")
    {
        return Err(RuntimeError::Device(format!(
            "{} lacks qualified DSA score/selection with PLOW_GQ_BATCH=1; rebuild the decode object",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn check_dsa_select_local(
    progs: &[DevProg],
    tensors: &[crate::asset::devblob::DevTensor],
    dec_ix: usize,
    arch: &str,
    tp8: bool,
) -> Result<()> {
    for (program, p) in progs.iter().enumerate() {
        for d in p
            .insts
            .iter()
            .filter(|d| d.op == DevOp::IndexSelect as u16 && d.i[4] != 0)
        {
            let err = || {
                RuntimeError::Device(
                "local DSA selection requires unpacked gfx942 TP8 decode rows 2/4/8/16/20, unpooled top2048 and row-sized operands".into())
            };
            // A token-batch body runs the decode form over its slot band inside a prefill-width
            // program: the band (the packet's block count) is the row count, not `p.t`.
            let rows = if p.token_batch_body {
                u32::from(d.blocks)
            } else {
                p.t
            };
            if arch != "gfx942"
                || !tp8
                || (program < dec_ix && !p.token_batch_body)
                || p.packed_prefill_only
                || !matches!(rows, 2 | 4 | 8 | 16 | 20)
                || u32::from(d.blocks) != rows
                || !(2048..=131072).contains(&d.i[0])
                || d.i[1..] != [2048, 0, 0, 1, 0, 0, 0]
                || d.fj != [0; 3]
                || [2, 3, 5, 6, 7]
                    .iter()
                    .any(|&i| d.t[i] != packet::dev::TENSOR_NONE16)
            {
                return Err(err());
            }
            let handles = [d.t[0], d.t[1], d.t[4]];
            if handles.iter().collect::<BTreeSet<_>>().len() != 3 {
                return Err(err());
            }
            for (handle, bytes) in handles.into_iter().zip([
                u64::from(rows) * 2048 * 4,
                u64::from(rows) * u64::from(d.i[0]) * 4,
                u64::from(rows) * 4,
            ]) {
                if tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes) {
                    return Err(err());
                }
            }
        }
    }
    Ok(())
}

pub(super) fn sparse_fp8(inst: &DevInst64) -> bool {
    matches!(
        DevOp::from_u16(inst.op),
        Some(DevOp::FlashMlaDecodeFp8 | DevOp::FlashMlaPrefillFp8)
    ) && inst.fj[1] != 0
}

pub(super) fn check_sparse_fp8_packet(
    progs: &[DevProg],
    tensors: &[crate::asset::devblob::DevTensor],
    arch: &str,
) -> Result<()> {
    for d in progs
        .iter()
        .flat_map(|p| &p.insts)
        .filter(|d| sparse_fp8(d))
    {
        let decode = d.op == DevOp::FlashMlaDecodeFp8 as u16;
        let rows = u64::from(if decode { d.i[0] } else { d.i[4] });
        let ctx = u64::from(d.i[2]);
        if arch != "gfx942"
            || d.i[1] != 8
            || d.i[3] != 0
            || d.i[5] != u32::MAX
            || d.fj[2] != 0
            || ctx < 2048
            || ctx > 81920
            || rows == 0
            || (decode && (rows > 20 || d.i[6] != 2048 || d.i[7] != 4))
            || (!decode
                && (d.i[0] != 1
                    || rows < 2048
                    || rows > 8192
                    || d.i[6] != d.i[2].min(16384)
                    || !mla_pf_v2_enabled()))
        {
            return Err(RuntimeError::Device(
                "sparse FP8 MLA requires qualified gfx942 QH8 geometry and V2 prefill routing"
                    .into(),
            ));
        }
        let slots = u64::from(d.i[0]);
        let selected = if decode {
            rows * 2048 * 4
        } else {
            (rows.div_ceil(8) * 4).div_ceil(256) * 256 + rows.div_ceil(8) * u64::from(d.i[6]) * 12
        };
        for (handle, bytes) in [
            (d.fj[1] - 1, selected),
            (u32::from(d.t[4]), slots * ctx * 512),
            (u32::from(d.t[5]), slots * ctx * 64 * 2),
            (u32::from(d.t[7]), slots * ctx * 4),
        ] {
            if handle >= u32::from(packet::dev::TENSOR_NONE16)
                || tensors.get(handle as usize).is_none_or(|t| t.bytes < bytes)
            {
                return Err(RuntimeError::Device(
                    "sparse FP8 MLA operand capacity is insufficient".into(),
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn check_sparse_fp8_object(
    syms: &[&str],
    path: &Path,
    progs: &[DevProg],
    decode: bool,
) -> Result<()> {
    let op = if decode {
        DevOp::FlashMlaDecodeFp8
    } else {
        DevOp::FlashMlaPrefillFp8
    };
    let marker = if decode {
        "plow_mla_sparse_fp8_decode_arm"
    } else {
        "plow_mla_sparse_fp8_prefill_arm"
    };
    if progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(|d| d.op == op as u16 && sparse_fp8(d))
        && !syms.contains(&marker)
    {
        return Err(RuntimeError::Device(format!(
            "{} lacks required sparse FP8 MLA marker {marker}",
            path.display()
        )));
    }
    Ok(())
}

/// `PLOW_DSA_PF_ARM=1` — the GATHERED (DSA sparse) V2 MLA-prefill arm.
///
/// Present iff the build axis is on, exactly like [`K3_ARMS_SYM`] and for a stronger reason: the
/// axis is OFF BY DEFAULT (instantiating the gathered body raised the flash object's spill
/// 98 -> 287 for every blob, sparse or not), so an object without the symbol has no gathered
/// body at all rather than merely predating one.
pub(super) const DSA_PF_ARM_SYM: &str = "plow_dsa_pf_arm";

/// Refuse a sparse MLA-prefill blob that cannot actually run sparse.
///
/// A `FlashMlaPrefill` (op 51) carrying a `t[7]` union table needs BOTH halves, and the failure
/// if either is missing is the same one and it is silent:
///
///   * **the V2 ROUTING**, because without `PLOW_MLA_PF_V2=1` the MLA segments are not split into
///     their own wave-class-4 segment and land on the 8-wave prefill kernel, whose
///     `exec_flash_mla_prefill` has no gathered arm for op 51. (It now `__builtin_trap()`s there
///     rather than running dense, but a load-time refusal says *why*, and a trap on a serving box
///     is a worse diagnostic than a message.)
///   * **the ARM ITSELF**, because the gathered instantiation is build-gated and off by default.
///
/// What makes this worth a hard refusal rather than a warning is the shape of the fallback. It is
/// not garbage a reader would notice and it is not slower-but-right: the dense arm never reads
/// `t[7]`, so the model gets FULL CAUSAL attention where it was trained sparse — finite, fluent,
/// and an answer to a different question. `mla_pf_v2` is passed in rather than read here so the
/// decision is a pure function of (object, blob, routing) and can be tested without a device.
pub(super) fn check_dsa_pf_arm(
    syms: &[&str],
    path: &Path,
    requires: &[String],
    mla_pf_v2: bool,
) -> Result<()> {
    if !requires.iter().any(|r| r == "PLOW_DSA_PF_ARM=1") {
        return Ok(());
    }
    if !mla_pf_v2 {
        return Err(RuntimeError::Device(
            "this packet carries a DSA union table on FlashMlaPrefill (op 51 t[7]) and REQUIRES \
             the V2 MLA-prefill routing: the 8-wave prefill kernel has no gathered arm for op 51 \
             and would run DENSE attention on a model trained sparse. Serve with \
             PLOW_MLA_PF_V2=1, or emit without PLOW_GLM_DSA_PF."
                .into(),
        ));
    }
    if !syms.contains(&DSA_PF_ARM_SYM) {
        return Err(RuntimeError::Device(format!(
            "packet/object MISMATCH: this packet requires PLOW_DSA_PF_ARM=1 but {} was compiled \
             without it (it does not advertise `{DSA_PF_ARM_SYM}`), so it carries no gathered V2 \
             body. Its dense arm does not read t[7] and AMD's dispatch does not trap — it would \
             run FULL CAUSAL attention on a model trained sparse and answer fluently and wrongly. \
             Rebuild the flash object with -DPLOW_DSA_PF_ARM=1.",
            path.display()
        )));
    }
    Ok(())
}

/// `PLOW_MLA_PF2_NOPE_ARM=1` — the zero-rope `d_flash_mla_prefill_v2<512, 0>` instantiation.
pub(super) const MLA_NOPE_ARM_SYM: &str = "plow_mla_pf2_nope_arm";

/// Refuse a NoPE (zero-rope) MLA-prefill blob against a flash object that predates the arm.
///
/// This is the load-time half of a gate that used to live in the segment-class pass as an
/// unconditional refusal — "NoPE MLA prefill cannot run on the four-wave V2 kernel". That was
/// true when no `<512, 0>` instantiation existed. Now one does, so the question is not what the
/// kernel FAMILY can do but what THIS OBJECT carries, and only the symbol table can answer it.
///
/// Unlike the DSA arm this one is default-on and free (the same body at DR=0: identical
/// VGPR/AGPR/LDS/spill, measured), so absence means the object is older, not that it declined
/// the cost. The device-side behaviour without the arm is a `__builtin_trap()` rather than a
/// silent wrong answer — deliberately, since a NoPE packet on a rope-only arm would otherwise
/// stage a `Krope` half that was never allocated — but a trap on a serving box is a far worse
/// diagnostic than a message naming the object and the rebuild.
pub(super) fn check_mla_nope_arm(syms: &[&str], path: &Path, requires: &[String]) -> Result<()> {
    if !requires.iter().any(|r| r == "PLOW_MLA_PF2_NOPE_ARM=1") {
        return Ok(());
    }
    if syms.contains(&MLA_NOPE_ARM_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object MISMATCH: this packet carries a NoPE (zero-rope) MLA prefill — op 51 with \
         i[3] bit 31 — but {} has no zero-rope V2 arm (it does not advertise \
         `{MLA_NOPE_ARM_SYM}`). Its <512,64> body would stage a Krope half this model does not \
         have; it traps instead of reading it, which is correct and useless as a diagnostic. \
         Rebuild the flash object from a tree that carries the arm (it is default-on and costs \
         no registers).",
        path.display()
    )))
}

/// The markers `runtime/amd/interp.hip` emits for the Gemma-4 MoE axes. [GEMMA4-MOE-AMD]
///
/// Two, not one, because the decode family (ops 61-72) and the grouped-prefill family (73-77,
/// 81/82) are separate build flags: the prefill half is a second full MFMA body and a decode-only
/// object must not carry it.
pub(super) const MOE_GEMMA_SYM: &str = "plow_moe_gemma_arms_1";
/// Marker for L2-DOMAIN DISPATCH (-DPLOW_L2_PLACE_DISPATCH). A placed blob carries per-domain
/// device queue windows, so an object without this axis mis-dispatches it SILENTLY.
/// Checked instead of the operator-asserted PLOW_L2_PLACE_DISPATCH env var.
pub(super) const L2_DISPATCH_SYM: &str = "plow_l2_place_dispatch_1";
pub(super) const GATE_HIER_SYM: &str = "plow_gate_hier_1";
pub(super) const MOE_GEMMA_PF_SYM: &str = "plow_moe_gemma_pf_arms_1";

/// The two halves of L2 placement never disagree silently, and the message names BOTH of them
/// plus the two ways out — the emit-side flag and the object-side flag are different names
/// living in different files, and a reader who only knows one of them cannot act on "lacks
/// `plow_l2_place_dispatch_1`".
pub(super) fn l2_pairing_refusal(path: &Path, phase: Phase) -> String {
    let (which, place_off, objects_on) = match phase {
        Phase::Decode => (
            "decode",
            "PLOW_L2_PLACE=0 at emit, which throws the decode win away with it",
            "scripts/build_gfx942.sh PLOW_L2HIER=1 or scripts/build_gfx950.sh PLOW_L2_PLACE=1 \
             — both are the default",
        ),
        Phase::Prefill | Phase::Flash => (
            "prefill",
            "no PLOW_L2_PLACE_PREFILL=1, which is the AMD default and leaves decode \
             placement on",
            "scripts/build_gfx942.sh PLOW_L2HIER_PF=1, which puts -DPLOW_L2_PLACE_DISPATCH on \
             the prefill rows (the flash rows already carry it); build_gfx950.sh passes it on \
             both under PLOW_L2_PLACE=1",
        ),
    };
    format!(
        "L2 PLACEMENT PAIRING: this blob's {which} programs are L2-placed (PLOW_L2_PLACE; \
         `build.json` records it under `l2_placement`), but {} was built WITHOUT \
         -DPLOW_L2_PLACE_DISPATCH. A placed program's `seg` is an L2 domain, not a wave class, \
         so this object would run every packet on the wrong domain — plausible output, inverted \
         locality, no error. Fix EITHER half: rebuild the objects with {objects_on}, or re-emit \
         the blob with {place_off}.",
        path.display()
    )
}

pub(super) fn check_gate_hier_object(
    syms: &[&str],
    path: &Path,
    phase: Phase,
    sched: Sched,
) -> Result<()> {
    if !syms.contains(&GATE_HIER_SYM) {
        return Ok(());
    }
    if phase != Phase::Decode || sched != Sched::GlobalQueue || !syms.contains(&L2_DISPATCH_SYM) {
        return Err(RuntimeError::Device(format!(
            "{} advertises `{GATE_HIER_SYM}`, but hierarchical gates are valid only for a \
             decode global-queue object carrying `{L2_DISPATCH_SYM}`",
            path.display()
        )));
    }
    Ok(())
}

/// What the two-level gate will actually do on this pairing.
///
/// ARMED and FIRING are different claims and only the second licenses a measurement. `armed`
/// is a property of the OBJECT alone (`plow_gate_hier_1`); the interpreter additionally
/// requires the BLOB to be L2-placed (`hier_base != 0`, which this engine derives from
/// `l2_domains`) and the stream entry to carry a per-domain slice count above 1
/// (`PLOW_SE_NPER`, set at emit only under placement). An armed object on an unplaced blob
/// compiles the hierarchy and never takes it — which is how every published Gemma-4-31B number
/// on this branch came to be measured with the feature inert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GateHierStatus {
    pub(super) armed: bool,
    pub(super) firing: bool,
    pub(super) domains: u32,
    pub(super) rendezvous: usize,
    pub(super) entries: usize,
}

impl GateHierStatus {
    /// The interpreter's own precondition, evaluated on the decode programs
    /// (`runtime/amd/interp.hip`, `h_on`).
    pub(super) fn of(progs: &[DevProg], dec_lo: usize, armed: bool) -> Self {
        let dec = &progs[dec_lo.min(progs.len())..];
        let domains = dec.iter().map(|p| p.l2_domains).max().unwrap_or(0);
        let mut rendezvous = 0usize;
        let mut entries = 0usize;
        for p in dec {
            for e in &p.gq_stream {
                entries += 1;
                let nper = (e.flags & packet::dev::SE_NPER_MASK) >> packet::dev::SE_NPER_SHIFT;
                if nper > 1 && e.flags & (packet::dev::SE_FINE | SE_XCTR) == 0 {
                    rendezvous += 1;
                }
            }
        }
        Self {
            armed,
            firing: armed && domains != 0 && rendezvous != 0,
            domains,
            rendezvous,
            entries,
        }
    }

    pub(super) fn verdict(&self) -> &'static str {
        match (self.armed, self.domains != 0, self.firing) {
            (true, true, true) => "FIRING — one L2 writeback+invalidate per XCD per packet",
            (true, true, false) => {
                "inert: the blob is L2-placed but no decode packet has more than one slice \
                 on a domain, so there is nobody to rendezvous with"
            }
            (true, false, _) => {
                "ARMED BUT INERT — the decode object carries PLOW_GATE_HIER and this blob is \
                 NOT L2-placed, which is its runtime precondition. Re-emit with \
                 PLOW_L2_PLACE=1 (the gfx942 default) to get the win"
            }
            (false, true, _) => {
                "off: the blob is L2-placed but the decode object was built without \
                 -DPLOW_GATE_HIER (scripts/build_gfx942.sh PLOW_L2HIER=1)"
            }
            (false, false, _) => "off: neither the decode object nor the blob asks for it",
        }
    }
}

/// Said once per rank, at load, because "compiled in" is not "running": the whole reason this
/// feature was measurable-but-unmeasured for a release cycle is that nothing printed the
/// difference.
pub(super) fn log_gate_hier_status(progs: &[DevProg], dec_lo: usize, armed: bool) {
    let st = GateHierStatus::of(progs, dec_lo, armed);
    tracing::info!(
        armed = st.armed,
        firing = st.firing,
        l2_domains = st.domains,
        rendezvous_entries = st.rendezvous,
        decode_queue_entries = st.entries,
        "L2 hierarchical gate: {}",
        st.verdict()
    );
}

/// Every opcode behind `#if PLOW_MOE_GEMMA` in `runtime/amd/interp.hip`.
pub(super) const MOE_GEMMA_OPS: &[DevOp] = &[
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
pub(super) const MOE_GEMMA_PF_OPS: &[DevOp] = &[
    DevOp::MoeRouterGemmaPf,
    DevOp::MoeAlignGemmaPf,
    DevOp::MoeGroupGluGemmaPf,
    DevOp::MoeGroupDownGemmaPf,
    DevOp::MoeCombineNormGemmaPf,
    DevOp::MoeGroupGluGemmaPfW8a8,
    DevOp::MoeGroupDownGemmaPfW8a8,
];

pub(super) fn first_op_in(progs: &[DevProg], set: &[DevOp]) -> Option<DevOp> {
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
pub(super) fn check_moe_gemma_arms(
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

/// The decode object's GEMV staging arena in halves (`interp.hip`, `plow_dec_stage_halves`).
pub(super) const DEC_STAGE_SYM: &str = "plow_dec_stage_halves";

/// Every decode op whose body stages `M*K` halves of `x` in the arena with NO global fallback.
///
/// `d_gemv_glu` / `d_gemv_qkvg` document the precondition as "plowc checks it"; plowc's model of
/// the arena is `hwspec`'s `decode_gemm_tile`, which on gfx942 is the OCC4 / DEC_SQUEEZE re-cut
/// and not what a default object is built at. `PLOW_DEC_STAGE_HALVES` lets the emitter state the
/// object's real arena, so the claim has to be CHECKED against the object that will run it —
/// otherwise a fused blob paired with a squeezed object stages off the end of LDS and decodes
/// fluent-but-wrong rows, the failure shape §6g-BATCH already recorded once.
pub(super) const DEC_STAGED_OPS: &[DevOp] = &[
    DevOp::GemvGlu,
    DevOp::GemvGluFp8,
    DevOp::GemvGluMxfp4,
    DevOp::GemvQkv,
    DevOp::GemvQkvFp8,
    DevOp::GemvQkvMxfp4,
];

/// Refuse a decode blob whose fused GEMV stages more of `x` than this object's arena holds.
///
/// `M` is `i[0]` and `K` is `i[2]` for every op in [`DEC_STAGED_OPS`] (`packet::slots`). An
/// object without the marker predates it and is left alone: it can only have been paired with a
/// blob emitted under the old conservative bound, which every arena satisfies.
pub(super) fn check_dec_stage_capacity(image: &[u8], path: &Path, progs: &[DevProg]) -> Result<()> {
    let Some(halves) = elf_symbol_u32(image, DEC_STAGE_SYM) else {
        return Ok(());
    };
    for p in progs {
        for inst in &p.insts {
            if !DEC_STAGED_OPS.iter().any(|&o| o as u16 == inst.op) {
                continue;
            }
            let need = u64::from(inst.i[0]) * u64::from(inst.i[2]);
            if need > u64::from(halves) {
                return Err(RuntimeError::Device(format!(
                    "packet/object STAGING MISMATCH: a fused decode GEMV (op {}) stages M*K = \
                     {need} halves of x, but {} has a {halves}-half arena. The fused bodies have \
                     no global-memory fallback, so this would write past LDS and decode \
                     fluent-but-wrong rows. Re-emit without PLOW_DEC_STAGE_HALVES, or build the \
                     decode object at the arena the blob was emitted for.",
                    inst.op,
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// The marker `runtime/amd/interp.hip` emits when it was compiled with `PLOW_FP8_KV=1`.
pub(super) const FP8_KV_SYM: &str = "plow_fp8_kv_1";

/// Every opcode that reaches an arm behind `#if PLOW_FP8_KV` — the fp8 half of the SWAP.
pub(super) const FP8_KV_OPS: &[DevOp] = &[
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
pub(super) const BF16_KV_OPS: &[DevOp] = &[
    DevOp::FlashDecode,
    DevOp::FlashPrefill,
    DevOp::FlashMlaDecode,
    DevOp::FlashMlaPrefill,
];

/// The first opcode in `progs` that reaches an arm on `side`, or `None`.
pub(super) fn required_kv_op(progs: &[DevProg], side: &[DevOp]) -> Option<DevOp> {
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
pub(super) fn check_kv_encoding(
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
pub(super) fn object_gemv_cap(syms: &[&str]) -> Option<u32> {
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
/// [`super::AmdEngine::load`] already refuses `batch > PLOW_GEMV_MAXM`, but that
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
pub(super) fn check_gemv_capacity(syms: &[&str], path: &Path, need: u32) -> Result<()> {
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

pub(super) fn check_xargmax_capacity(syms: &[&str], path: &Path, need: u32) -> Result<()> {
    if need <= 32 || syms.contains(&XARGMAX_B128_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object XArgmax MISMATCH: decode needs {need} rows, but {} does not advertise \
         `{XARGMAX_B128_SYM}`. A pre-B128 object writes only rows 0..32 and would leave the \
         remaining sampled ids stale. Rebuild the decode object from the current tree.",
        path.display()
    )))
}

/// Is the V2 MLA-prefill routing enabled (`PLOW_MLA_PF_V2!=0`)?
///
/// Enabled by default: it moves `FlashMlaPrefill` segments onto a 4-wave object, whose
/// full-column-wave kernel (`d_flash_mla_prefill_v2`) needs the 512-register budget the
/// 8-wave interpreter cannot give. gfx950 prefers the dedicated scratch-free V2+SV object;
/// if it is unavailable, the capability-checked general flash object remains the exact fallback.
pub fn mla_pf_v2_enabled() -> bool {
    crate::config::RuntimeConfig::get().amd.mla_pf_v2
}

/// The V2 arms' marker symbols (see `interp.hip`).
pub(super) const MLA_PF_V2_SYM: &str = "plow_mla_pf_v2_arm_1";
pub(super) const MLA_PF_V2_FP8_SYM: &str = "plow_mla_pf_v2_fp8_arm_1";
pub(super) const MLA_PF_V2_SV_RAW_SYM: &str = "plow_mla_pf_v2_sv_raw_1";

pub(super) fn resolve_flash_object_load<T>(load: Result<T>, required: bool) -> Result<Option<T>> {
    match load {
        Ok(object) => Ok(Some(object)),
        Err(error) if required => Err(error),
        Err(_) => Ok(None),
    }
}

pub(super) fn check_mla_v2_sv_raw_symbols(
    syms: &[&str],
    path: &Path,
    needs_l2: bool,
) -> Result<()> {
    for marker in [MLA_PF_V2_SYM, MLA_PF_V2_SV_RAW_SYM] {
        if !syms.contains(&marker) {
            return Err(RuntimeError::Device(format!(
                "raw MLA V2 object {} lacks required marker `{marker}`",
                path.display()
            )));
        }
    }
    if syms.contains(&MLA_PF_V2_FP8_SYM) || syms.contains(&PACKED_PREFILL_MLA_FLASH_SEG_SYM) {
        return Err(RuntimeError::Device(format!(
            "raw MLA V2 object {} carries an fp8 or packed-prefill consumer arm",
            path.display()
        )));
    }
    if needs_l2 && !syms.contains(&L2_DISPATCH_SYM) {
        return Err(RuntimeError::Device(format!(
            "raw MLA V2: {}",
            l2_pairing_refusal(path, Phase::Flash)
        )));
    }
    Ok(())
}
