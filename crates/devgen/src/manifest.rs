//! `build.json` — what a compiled packet REQUIRES of the object that runs it.
//!
//! The packet (what `plowc` emits) and the interpreter object (what `nvcc -D…`
//! produces) were two independent sources of truth and nothing checked they
//! agreed. Five separate measured failures came out of exactly that gap: a
//! decode asset built `GV_MM_MAX=16` and served at B=8 (−19.4% at 131k), a
//! `PLOW_NV_FA_GF_FULL=4` that reached one object and not its sibling (1.48x
//! left on the floor), an fp8-KV prefill object that hardcoded the slow staging
//! arm (5.4x on a 127k prefill), a `PLOW_W8A8=1` packet against a cubin without
//! the arm (`__trap()` → `CUDA_ERROR_LAUNCH_FAILED`, which reads as a driver
//! bug), and an all-layer fp8-KV asset benchmarked against a mixed-KV baseline.
//! This file is the shared fact both sides can be checked against.
//!
//! ## Everything here is derived from the EMITTED INSTRUCTION STREAM
//!
//! Not from the emitter's intent — not from the `PLOW_FP8_KV` env var, not from
//! the `Cfg` — from [`Model::progs`]`[..].insts`. An emitter flag says what was
//! *asked for*; the instruction stream is what a packet actually *contains*, and
//! only the second one is what the object has to be able to run. Deriving from
//! intent would reintroduce the drift this exists to kill, one level up.
//!
//! ## Arms, not opcodes
//!
//! One opcode can reach several instantiated bodies: `FlashDecode` and
//! `FlashDecodeFp8` are templated on head dim (256 vs 512), which is an
//! INSTRUCTION FIELD (`i[6]`), not an opcode. An arm set derived from opcodes
//! alone is wrong in both directions — it keeps bodies nothing dispatches to and
//! it can drop one that a runtime field selects. So the unit here is an [`Arm`]:
//! opcode plus the static shape that selects the template instantiation.
//!
//! ## Arch-agnostic
//!
//! The manifest names OPCODES, SHAPES and RULES. It never names a `-D` flag.
//! Rendering the neutral facts into a toolchain's flags is a BACKEND's job
//! ([`Backend`]) — `nvcc → .cubin` today, `hipcc → .hsaco` (runtime/amd/) later.
//! Keeping the flag vocabulary out of the schema is what makes the AMD backend a
//! backend rather than a redesign.

use std::collections::{BTreeMap, BTreeSet};

use packet::dev::{DevOp, GEMM_WIDE_C8_TAG};
use packet::devbuild::{Model, Program};
use serde_json::{json, Map, Value};

/// The packet's opcode-name → `DevOp` lookup. `DevOp` has no `from_u16`, and the
/// discriminant range has holes, so scan the hand-maintained `ALL` table — which
/// `dev_abi.rs` already gates against the enum, so it cannot go stale.
fn op_of(code: u16) -> Option<DevOp> {
    DevOp::ALL.iter().copied().find(|o| *o as u16 == code)
}

/// The Rust spelling of an opcode (`c_name` minus the `PLOW_DOP_` prefix would
/// lose the CamelCase the manifest is nicer to read in, so use `Debug`).
fn op_name(op: DevOp) -> String {
    format!("{op:?}")
}

/// One instantiated kernel body: an opcode plus the static shape that selects
/// which template instantiation the dispatch reaches.
///
/// Rendered as `"FlashDecode/hd512"` — one string, because these lists repeat
/// per program and a list of objects triples the file for no added meaning.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Arm {
    pub op: String,
    /// Head dim for the flash family (`i[6]`). `None` for ops with one body.
    pub hd: Option<u32>,
    /// Normalized instruction-immediate-selected body variant. Kept separate from
    /// `hd` so schema-1 consumers retain the existing flash arm spelling.
    pub variant: Option<String>,
}

impl Arm {
    pub fn key(&self) -> String {
        let mut key = match self.hd {
            Some(hd) => format!("{}/hd{hd}", self.op),
            None => self.op.clone(),
        };
        if let Some(variant) = &self.variant {
            key.push('/');
            key.push_str(variant);
        }
        key
    }
}

/// Which shape field (if any) selects a template instantiation for this opcode.
///
/// Flash and head-normalization are templated on an instruction field today.
/// The GEMM tile variants are separate OPCODES
/// (`Gemm`/`GemmMed`/`GemmSmall`), so the opcode already carries them and adding a
/// second key would split one body into several phantom arms.
///
/// THE SLOT IS NOT UNIFORM ACROSS THE FAMILY, and reading the wrong one is silent: the
/// arm comes out `Some(0)`, matches nothing the object declares, and the coverage check
/// passes because it is comparing a constant against a constant.
///
///   * `FlashPrefill` / `FlashDecode` (and their fp8 twins) carry HD in `i[6]`
///     (`runtime/amd/interp.hip`, `exec_flash_decode`: `const unsigned hd = in->i[6];`).
///   * `FlashMerge` carries it in **`i[3]`** — kernel `runtime/amd/interp.hip:1119`/`:1315`
///     (`if (in->i[3] == 128) d_flash_merge<128>(...)`), emitters
///     `crates/devgen/src/lib.rs` (`d.i[3] = hd`) and `crates/devgen/src/mla.rs:4626`
///     (`i.i[3] = c.attn_head_dim`). `i[6]` is never assigned on a `FlashMerge` packet.
fn arm_of(op: DevOp, i: &[u32; 8]) -> Arm {
    let hd = match op {
        DevOp::FlashMerge => Some(i[3]),
        DevOp::HeadNormRope | DevOp::HeadNormRopeFp8 => Some(i[2]),
        DevOp::FlashPrefill
        | DevOp::FlashPrefillFp8
        | DevOp::FlashDecode
        | DevOp::FlashDecodeFp8 => Some(i[6]),
        _ => None,
    };
    let variant = match op {
        DevOp::KdaChunkPrepare | DevOp::KdaChunkIntra => Some(format!("d{}", i[2])),
        DevOp::KdaChunkWu | DevOp::KdaChunkCarry => {
            Some(format!("d{}{}", i[2], if i[4] != 0 { "_qpre" } else { "" }))
        }
        DevOp::KdaDecodeFused => Some(format!("abi{}", i[7])),
        DevOp::KdaStateStep | DevOp::KdaStateStepG | DevOp::KdaConvStateStepG => {
            Some(format!("flags{:x}", i[4]))
        }
        DevOp::GemmWide if i[7] == GEMM_WIDE_C8_TAG => Some("tile128x384x64".into()),
        _ => None,
    };
    Arm {
        op: op_name(op),
        hd,
        variant,
    }
}

/// The arm set of one program, or of one SEGMENT of one program.
///
/// `seg` is a partition key like `bucket`, not a different kind of thing: with
/// `PLOW_NV_SEGMENTS=1` the host relaunches the interpreter once per segment
/// (`prog.cur_seg`), so each segment CAN carry its own register/occupancy
/// profile — that is why `_seg` / `_gemm` / `_gemm_bn64` exist as objects at all.
/// `None` = the program is single-segment (which is every program `plowrt serve`
/// can currently reach; see the `check_coarse_single_segment` note in
/// `crates/plowrt/src/exec/gpu.rs`).
#[derive(Clone, Debug)]
pub struct ProgramArms {
    /// Stable index in `Model::progs`.  Segment rows with the same bucket are
    /// otherwise ambiguous (ordinary and packed ladders can share a width).
    pub program: usize,
    pub kind: &'static str,
    pub packed_prefill_only: bool,
    /// Prefill chunk rows, or decode batch — the `T` the program was compiled for.
    pub t: u32,
    pub seg: Option<u32>,
    pub arms: BTreeSet<Arm>,
    /// Instruction count, so the ceiling attribution below can say how big the
    /// program that owns a ceiling actually is.
    pub insts: usize,
}

/// Per-program and per-segment arm sets, plus the union the object must compile.
fn program_arms(m: &Model) -> Vec<ProgramArms> {
    let mut out = Vec::new();
    // Prefill buckets precede an ascending decode-rung suffix. Use the same canonical
    // boundary as the runtime; treating only the last program as decode makes a specialized
    // prefill object absorb every lower decode rung.
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    for (pi, p) in m.progs.iter().enumerate() {
        let kind = if pi >= dec_lo { "decode" } else { "prefill" };
        let encoded_t = m.prog_t.get(pi).copied().unwrap_or(0);
        let t = packet::devbuild::program_rows(encoded_t);
        for (seg, arms) in segment_arms(p) {
            out.push(ProgramArms {
                program: pi,
                kind,
                packed_prefill_only: packet::devbuild::is_packed_prefill_program(encoded_t),
                t,
                seg,
                insts: arms.1,
                arms: arms.0,
            });
        }
    }
    out
}

/// Split one program's arms by segment. A single-segment program yields exactly
/// one entry with `seg: None`, so the unsegmented case reads as it always did.
#[allow(clippy::type_complexity)]
fn segment_arms(p: &Program) -> Vec<(Option<u32>, (BTreeSet<Arm>, usize))> {
    // `StreamEnt.seg` is always the ordered kernel-family segment. L2 placement
    // multiplies `gq_seg_ofs` by the physical-domain count, so derive the host
    // dimension from the stream rather than mistaking XCD queues for segments.
    let n_seg = p
        .stream
        .iter()
        .map(|e| e.seg as usize + 1)
        .max()
        .unwrap_or(1);
    if n_seg <= 1 || p.gq_stream.is_empty() {
        let mut arms = BTreeSet::new();
        for inst in &p.insts {
            if let Some(op) = op_of(inst.op) {
                arms.insert(arm_of(op, &inst.i));
            }
        }
        return vec![(None, (arms, p.insts.len()))];
    }
    let mut out = Vec::new();
    let domains = if p.l2_domains != 0
        && p.gq_seg_ofs.len().saturating_sub(1) == n_seg * p.l2_domains as usize
    {
        p.l2_domains as usize
    } else {
        1
    };
    for s in 0..n_seg {
        let w = s * domains;
        let (a, b) = (p.gq_seg_ofs[w] as usize, p.gq_seg_ofs[w + domains] as usize);
        let mut arms = BTreeSet::new();
        let mut seen = BTreeSet::new();
        for ent in &p.gq_stream[a.min(p.gq_stream.len())..b.min(p.gq_stream.len())] {
            let inst = &p.insts[ent.inst as usize];
            seen.insert(ent.inst);
            if let Some(op) = op_of(inst.op) {
                arms.insert(arm_of(op, &inst.i));
            }
        }
        out.push((Some(s as u32), (arms, seen.len())));
    }
    out
}

/// `next_pow2` — the `GV_MM_MAX` rule's arithmetic. `0`/`1` ⇒ 1.
fn next_pow2(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        1u32 << (32 - (n - 1).leading_zeros())
    }
}

/// Shapes read off the instruction stream. Every field here is an instruction
/// operand of an op that is actually present — nothing is inferred from config.
#[derive(Default, Debug)]
struct Shapes {
    /// Head dims the flash family is instantiated at, ascending.
    hd: BTreeSet<u32>,
    /// KV head counts, ascending. A GQA model has one per attention regime.
    kv_heads: BTreeSet<u32>,
    /// `n_head / n_kv_head` over the FULL-attention (largest hd) decode sites.
    gqa: u32,
    /// KV heads on the full-attention sites — the `GF_FULL = gqa` rule's guard.
    full_kv_heads: u32,
    /// Decode batch: `n_batch` on the decode program's flash sites.
    decode_batch: u32,
    /// hd → "bf16" | "e4m3", from which flash opcode reads that hd.
    kv_dtype: BTreeMap<u32, &'static str>,
    /// Largest prefill bucket = the largest chunk the runtime can submit.
    max_chunk: u32,
    prefill_buckets: Vec<u32>,
    /// MoE weight encodings present on the expert ops. A SET, not a scalar — if a packet ever
    /// carried two, that is a mixed-precision run and the manifest is where it becomes visible
    /// rather than a number nobody can explain. NOTE the slot differs by phase: `i[3]` on the
    /// PREFILL grouped ops (85/86), `i[6]` on the DECODE expert ops (45/46/48/49), because those
    /// four already used `i[3]` for `n_exp`. Reading the wrong slot here would report `n_exp` as an
    /// encoding, so the two are collected separately and deliberately.
    moe_enc: BTreeSet<u32>,
    /// Any w4a16 MXFP4 projection opcode present (91/92/93).
    mxfp4_proj: bool,
    /// Any prefill DOWN/combine with `i[7]=1` — bf16 `part` scatter (PLOW_MOE_PF_PART16).
    moe_pf_part16: bool,
    /// Any prefill GLU with `i[7]=1` — fp8 gathered activations (PLOW_MOE_PF_A8).
    moe_pf_a8: bool,
    /// Any decode one-shot XReduce with `i[7]!=0` — the K3 latent MoeCombine folded into the
    /// tagged publish (PLOW_XR_COMBINE_FOLD). An object without the arm publishes the unwritten
    /// plain slot: finite, stale, wrong.
    xr_combine_fold: bool,
    /// Any `KdaStateStepG` with flags bit 2 — the f_b GEMV folded into the step's prologue
    /// (PLOW_KDA_FB_FOLD). An object without the arm reads `f_a` as the gate logits.
    kda_fb_fold: bool,
    /// Any `KdaStateStepG` with flags bit 3 — the Conv3+StepG+GatedNorm chain as one packet
    /// (PLOW_KDA_DECODE_FUSED_ARM). An object without the arm runs the recurrence on the RAW
    /// projections and reads the descriptor as `A_log`.
    kda_decode_fused_arm: bool,
    /// Any prefill DOWN with `i[4]!=0` — the fused 86->87 decomposition (PLOW_MOE_PF_ATOMIC):
    /// t[0] is a [T,H] f32 accumulator the DOWN epilogue atomically adds into, NOT the
    /// [T*k,H] `part` scatter. An object without the arm would overrun it k-fold.
    moe_pf_atomic: bool,
    /// Any prefill DOWN with `i[5]!=0` — the DETERMINISTIC fused decomposition (PLOW_MOE_PF_DET):
    /// t[0] is a [T,H] **f64** fixed-point accumulator. Same overrun class as `moe_pf_atomic`,
    /// and additionally op 87 would read f64 bytes as f32 without the arm.
    moe_pf_det: bool,
    /// Compiler-declared replicated-input expert-parallel boundaries as
    /// `(degree, experts, full_intermediate_width)` tuples.
    moe_prefill_ep: BTreeSet<(u32, u32, u32)>,
    /// Any DENSE FlashMlaPrefill with `(i[6] & 0xff) > 1` — the causal KV-split partial
    /// layout (PLOW_MLA_PF_NS). The sparse GATHER arm reuses `i[6]` whole as `cap`,
    /// disambiguated by the union table in `t[7]`.
    mla_pf_ns: bool,
    /// Any dense `FlashMlaPrefill` with `i[6]` bit 8 set and no t7 — the W_ofold fusion
    /// (PLOW_GLM_OFOLD): normalized-bf16 flash epilogue + fused o-GEMM. Exclusive with the
    /// KV-split on a packet (the fold consumes the un-split l), so the two live in one
    /// bitfield: low 8 bits = ns, bit 8 = ofold.
    glm_ofold: bool,
    /// Any dense `FlashMlaDecode` (op 50) carrying a `t[7]` — the DECODE q-rope fold
    /// (PLOW_GLM_FUSE_ROPE): t7 = cos table, i6 = sin handle, and `t[3]` is the RAW q_rope
    /// projection. A pre-arm object ignores both and stages `t[3]` verbatim, i.e. it feeds the
    /// flash an UNROPED query and returns finite, fluent, wrong tokens. Refuse at load.
    glm_fuse_rope: bool,
    /// Any `QuantFp8` carrying a `t[3]` — the T11 GLU-into-quant fold (PLOW_T11_GLUQUANT):
    /// t3=gate, t4=up, i2=act, and `t[1]` becomes an OUTPUT. The emitter DELETES the `Glu`
    /// packet when it folds, so an object whose QUANT_FP8 arm ignores t3/t4 quantizes an
    /// activation nothing produced — garbage FFN output, wrong KV cache, fluent wrong tokens.
    /// That is the shape the AMD runtime shipped in; refuse the pairing at load.
    quant_glu_fold: bool,
    /// Any `GemvQkv` (op 22) carrying a `t[7]` — the DECODE q-norm fold
    /// (PLOW_GLM_FUSE_QNORM): t7 = gamma, f0 = eps, and `t[1]` is the RAW pre-norm activation.
    /// A pre-arm object ignores both and stages `t[1]` verbatim, i.e. it runs the projection
    /// over an UNNORMED row and returns finite, fluent, wrong tokens. Refuse at load.
    glm_fuse_qnorm: bool,
    /// Optional linear biases carried by the instruction stream. Plain GEMM/GEMV use `t7`;
    /// fused QKV uses the three demoted handles in `i5/i6/i7`.
    linear_bias: bool,
    /// HD64 HeadNormRope using the explicit NeoX half-split pairing mode.
    rope_half_hd64: bool,
    /// FlashMerge carrying an attention-sink vector in `t3`.
    attention_sinks: bool,
    /// A GEMV instruction in a prefill program (currently the M=1 lm_head).
    prefill_gemv: bool,
    /// Opcode names present, for the encoding-aware corrections below. Kept as names because that
    /// is what `features` keys on, and the two must not disagree.
    ops_present: BTreeSet<String>,
}

/// Was this opcode emitted anywhere in the packet?
fn union_has(s: &Shapes, name: &str) -> bool {
    s.ops_present.contains(name)
}

fn shapes(m: &Model) -> Shapes {
    let mut s = Shapes::default();
    // Every program from `dec_lo` on is a DECODE RUNG, not a prefill bucket. Without a
    // ladder that is the last program alone, i.e. exactly the `pi == last` this replaced.
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    for (pi, p) in m.progs.iter().enumerate() {
        let decode = pi >= dec_lo;
        if !decode
            && !packet::devbuild::is_packed_prefill_program(m.prog_t.get(pi).copied().unwrap_or(0))
        {
            s.prefill_buckets.push(packet::devbuild::program_rows(
                m.prog_t.get(pi).copied().unwrap_or(0),
            ));
        }
        for inst in &p.insts {
            let Some(op) = op_of(inst.op) else { continue };
            s.ops_present.insert(op_name(op));
            match op {
                DevOp::XReduce if inst.i[7] != 0 => s.xr_combine_fold = true,
                DevOp::KdaStateStepG if inst.i[4] & 12 != 0 => {
                    s.kda_fb_fold |= inst.i[4] & 4 != 0;
                    s.kda_decode_fused_arm |= inst.i[4] & 8 != 0;
                }
                // `i0=n_batch i1=n_head i2=n_kv_head … i6=hd`
                DevOp::FlashDecode | DevOp::FlashDecodeFp8 => {
                    let (hd, nh, kvh, nb) = (inst.i[6], inst.i[1], inst.i[2], inst.i[0]);
                    s.hd.insert(hd);
                    s.kv_heads.insert(kvh);
                    s.kv_dtype.insert(
                        hd,
                        if op == DevOp::FlashDecodeFp8 {
                            "e4m3"
                        } else {
                            "bf16"
                        },
                    );
                    if decode {
                        s.decode_batch = s.decode_batch.max(nb);
                        // The FULL-attention regime is the largest head dim
                        // (hd512 on Gemma-4; sliding is hd256).
                        if kvh > 0 && hd >= s.hd.iter().copied().max().unwrap_or(0) {
                            s.gqa = nh / kvh;
                            s.full_kv_heads = kvh;
                        }
                    }
                }
                // `… i2=n_head i3=n_kv_head … i6=hd`
                DevOp::FlashPrefill | DevOp::FlashPrefillFp8 => {
                    s.hd.insert(inst.i[6]);
                    s.kv_heads.insert(inst.i[3]);
                    s.kv_dtype
                        .entry(inst.i[6])
                        .or_insert(if op == DevOp::FlashPrefillFp8 {
                            "e4m3"
                        } else {
                            "bf16"
                        });
                }
                // The encoding slot is PHASE-DEPENDENT. Prefill grouped ops carry `n_exp` in i[2],
                // so the encoding took i[3]; the decode expert ops predate the field and already use
                // i[3] for `n_exp`, so theirs is i[6]. Reading i[3] on a decode op would report the
                // expert COUNT as an encoding.
                DevOp::MoeGroupGluPf | DevOp::MoeGroupDownPf => {
                    s.moe_enc.insert(inst.i[3]);
                    if inst.i[6] > 1 {
                        s.moe_prefill_ep.insert((
                            inst.i[6],
                            inst.i[2],
                            if op == DevOp::MoeGroupGluPf {
                                inst.i[0]
                            } else {
                                inst.i[1]
                            },
                        ));
                    }
                    // i[7] carries the activation-side arms: a8 on GLU, part16 on DOWN.
                    if inst.i[7] != 0 {
                        if op == DevOp::MoeGroupGluPf {
                            s.moe_pf_a8 = true;
                        } else {
                            s.moe_pf_part16 = true;
                        }
                    }
                    // i[4] on DOWN is the fused decomposition's log2(k)+1 (PLOW_MOE_PF_ATOMIC).
                    if op == DevOp::MoeGroupDownPf && inst.i[4] != 0 {
                        s.moe_pf_atomic = true;
                    }
                    // i[5] on DOWN is the deterministic arm's log2(k)+1 (PLOW_MOE_PF_DET).
                    if op == DevOp::MoeGroupDownPf && inst.i[5] != 0 {
                        s.moe_pf_det = true;
                    }
                }
                DevOp::QuantFp8 => {
                    if inst.t[3] != packet::TENSOR_NONE {
                        s.quant_glu_fold = true;
                    }
                }
                DevOp::MoeCombinePf => {
                    if inst.i[7] != 0 {
                        s.moe_pf_part16 = true;
                    }
                }
                // Dense V2 arm i[6] bitfield (sparse GATHER packets carry i[6]=cap but
                // always with t7 = the union table — the t7 test is the discriminator):
                // low 8 bits = causal KV-split ns, bit 8 = the W_ofold epilogue.
                // Op 50's t[7] is the q-rope fold and nothing else: the GATHER twin (54) and the
                // fp8 twin (109) are separate opcodes, and dense op 50 never had a t7 before the
                // fold. So presence IS the feature, with no bitfield to decode — unlike op 51
                // below, whose i[6] packs two.
                DevOp::FlashMlaDecode => {
                    if inst.t[7] != packet::TENSOR_NONE {
                        s.glm_fuse_rope = true;
                    }
                    // `i[0] = n_batch` on the MLA decode flash, exactly as on the dense-GQA
                    // twin above. Without this a batched GLM blob records `decode_batch: 1`
                    // and `gv_mm_max: 1`, and the object recipe built from that manifest is
                    // too narrow — `check_gemv_capacity` then refuses at load, or worse, an
                    // MM=1 object serves one correct row of B.
                    if decode {
                        s.decode_batch = s.decode_batch.max(inst.i[0]);
                    }
                }
                DevOp::FlashGatherDecode | DevOp::FlashMlaDecodeFp8 => {
                    if decode {
                        s.decode_batch = s.decode_batch.max(inst.i[0]);
                    }
                }
                // Op 22's t[7] is the q-norm fold and nothing else: op 108 `GemvQkvg`'s t7 is
                // `g_out` on a DIFFERENT opcode, and ops 114/115 put their scale rows in
                // i5/i6/i7. Dense op 22 never had a t7 before the fold, so presence IS the
                // feature — no bitfield to decode.
                DevOp::GemvQkv => {
                    if inst.t[7] != packet::TENSOR_NONE {
                        s.glm_fuse_qnorm = true;
                    }
                    if inst.i[5..8].iter().any(|&h| h != 0) {
                        s.linear_bias = true;
                    }
                }
                DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall | DevOp::Gemv => {
                    if inst.t[7] != packet::TENSOR_NONE {
                        s.linear_bias = true;
                    }
                    if op == DevOp::Gemv && !decode {
                        s.prefill_gemv = true;
                    }
                }
                DevOp::HeadNormRope => {
                    if inst.i[2] == 64 && inst.i[5] == packet::dev::ROPE_PAIR_HALF {
                        s.rope_half_hd64 = true;
                    }
                }
                DevOp::FlashMerge => {
                    if inst.t[3] != packet::TENSOR_NONE {
                        s.attention_sinks = true;
                    }
                }
                DevOp::FlashMlaPrefill => {
                    if (inst.i[6] & 0xff) > 1 && inst.t[7] == packet::TENSOR_NONE {
                        s.mla_pf_ns = true;
                    }
                    if (inst.i[6] >> 8) & 1 == 1 && inst.t[7] == packet::TENSOR_NONE {
                        s.glm_ofold = true;
                    }
                }
                DevOp::MoeExpertGluFp8Blk
                | DevOp::MoeExpertDownFp8Blk
                | DevOp::MoeGroupGluFp8Blk
                | DevOp::MoeGroupDownFp8Blk => {
                    s.moe_enc.insert(inst.i[6]);
                }
                // EVERY mxfp4 tile rung, not just the 256x256 one. This classifier decides
                // `mxfp4_weights`, which decides which OBJECT the host loads
                // (`scripts/gfx950_objects.py:150`). Before the prefill GEMM became
                // tile-selected there was exactly one fp4 prefill opcode and listing it was
                // the same as listing the family; now a packet whose only fp4 GEMM is
                // `GemmSmallMxfp4` would be classified bf16, load `interp_prefill`, and hit
                // that object's silent `default:` — an untouched output buffer read as a
                // result. §4's shape, reached by ADDING an arm rather than by forgetting one.
                DevOp::GemvMxfp4
                | DevOp::GemvGluMxfp4
                | DevOp::GemvQkvMxfp4
                | DevOp::GemmGluMxfp4
                | DevOp::GemmMxfp4
                | DevOp::GemmMedMxfp4
                | DevOp::GemmSmallMxfp4
                | DevOp::GemmWideMxfp4
                | DevOp::GemmC5Mxfp4 => {
                    s.mxfp4_proj = true;
                }
                _ => {}
            }
        }
    }
    s.max_chunk = s.prefill_buckets.iter().copied().max().unwrap_or(0);
    s
}

// Every opcode whose presence means "this packet needs an fp8 WEIGHT arm compiled in".
//
// ONE list, used by both the provisional classification and the encoding-field correction
// below. They were two hand-maintained copies of the same six names, and the tile-inventory
// campaign added `GemmWideFp8`/`GemmC5Fp8` — an fp8 packet whose only GEMM took one of the
// new rungs would have been classified `fp8_weights: false` by BOTH copies, so the host
// would load `interp_prefill` (no fp8 arm) and the GEMM would hit its silent `default:`.
// That is the §4 shape reached by adding an arm; the single list is what makes the next
// rung a one-line edit instead of two that can disagree.
const FP8_WEIGHT_OPS: &[&str] = &[
    "GemvFp8",
    "GemvQkvFp8",
    "GemvGluFp8",
    "GemmFp8",
    "GemmMedFp8",
    "GemmSmallFp8",
    "GemmWideFp8",
    "GemmC5Fp8",
    "GemmGluFp8",
    "GemvFp8Blk",
    "DenseGluFp8Blk",
    // The DENSE PREFILL block-fp8 GEMM. It has to be here even though it is dispatched
    // UNCONDITIONALLY in every prefill object (so no object choice actually turns on it), because
    // this list also drives the `encoding_features` CORRECTION below: a packet whose only fp8
    // weights are dense — block-fp8 linears with no block-fp8 MoE, so `moe_enc` never sees 1 —
    // takes the branch that recomputes `fp8_weights` from THIS list alone. Omitted, such a manifest
    // would record `weight_enc: "bf16"` while the stream is fp8: a manifest that says the opposite
    // of what it contains. GLM's stacked blob is unaffected either way (its decode `GemvFp8Blk`
    // already sets the flag), which is exactly why the omission would have been invisible.
    "GemmFp8Blk",
];

/// Neutral capability facts. Presence of an ARM implies the feature — the env
/// knob that produced it is not consulted.
fn features(union: &BTreeSet<Arm>) -> Map<String, Value> {
    let has = |n: &str| union.iter().any(|a| a.op == n);
    let mut f = Map::new();
    // `FlashMlaDecodeFp8` / `FlashMlaPrefillFp8` are here for the reader-only case. An MLA
    // fp8-KV packet writes its latent with `HeadNormRopeFp8` and so already trips the clause
    // above, but a packet that only READS a pre-quantized cache (a resumed session, a prefix
    // shared between requests) has no writer in its stream, and the object it needs is exactly
    // the same one.
    f.insert(
        "fp8_kv".into(),
        json!(
            has("FlashDecodeFp8")
                || has("HeadNormRopeFp8")
                || has("FlashMlaDecodeFp8")
                || has("FlashMlaPrefillFp8")
        ),
    );
    // The `*Fp8Blk` family is fp8 WEIGHTS too — DeepSeek's [128,128] `weight_scale_inv` grid rather
    // than a per-channel scale, but the object still needs an fp8 weight arm compiled. Omitting them
    // reported `fp8_weights: false` for a block-fp8 Kimi/GLM/DeepSeek packet, which is the manifest
    // saying the opposite of what the stream contains.
    //
    // NOTE this is only PROVISIONAL: once the MoE ops took a runtime encoding field, their OPCODE
    // NAME stopped implying the encoding — `MoeExpertGluFp8Blk` carrying `i[6]=2` is an MXFP4 op
    // with an fp8-era name. `encoding_features` corrects this from the encoding set. The two-step
    // exists because that is exactly the drift this file was written to catch, in miniature: a name
    // that used to be a fact became a label.
    f.insert(
        "fp8_weights".into(),
        json!(
            FP8_WEIGHT_OPS.iter().any(|k| has(k)) || union.iter().any(|a| a.op.ends_with("Fp8Blk"))
        ),
    );
    // w8a8 is the per-row ACTIVATION quant: `QuantFp8` exists only on that path.
    f.insert("w8a8".into(), json!(has("QuantFp8")));
    f.insert(
        "qwen_gdn".into(),
        json!(union.iter().any(|a| a.op.starts_with("Qwen"))),
    );
    f.insert(
        "moe".into(),
        json!(union.iter().any(|a| a.op.starts_with("Moe"))),
    );
    f.insert(
        "mla".into(),
        json!(
            has("FlashMlaDecode")
                || has("FlashMlaPrefill")
                || has("FlashMlaDecodeFp8")
                || has("FlashMlaPrefillFp8")
                || has("MlaMergeFold")
        ),
    );
    f.insert("mamba".into(), json!(has("Mamba2Scan")));
    f.insert(
        "tensor_parallel".into(),
        json!(union.iter().any(|a| a.op.starts_with('X'))),
    );
    // "prefill" = the packet carries a prefill program, which for the MLA family means the LATENT
    // flash, not the dense one. Keying only on FlashPrefill reported `prefill: false` for a Kimi
    // packet whose buckets are the whole reason the object needs PLOW_MLA_PREFILL=1.
    f.insert(
        "prefill".into(),
        json!(
            has("FlashPrefill")
                || has("FlashPrefillFp8")
                || has("FlashMlaPrefill")
                || has("FlashMlaPrefillFp8")
                || has("FlashGatherPrefill")
        ),
    );
    // The FFN half of an MLA-family prefill is a SEPARATE object axis (PLOW_MOE_PREFILL): the
    // grouped-expert GEMM is a second full MFMA body in a bucket already at the register cliff, so a
    // Gemma or attention-only object must not compile it.
    f.insert(
        "moe_prefill".into(),
        json!(union.iter().any(|a| a.op.ends_with("Pf")) || has("MoeAlignPf")),
    );
    f
}

/// Encoding facts that live on an instruction FIELD rather than an opcode.
///
/// The MoE weight encoding is `i[3]` on ops 85/86 precisely so a precision change is not a re-emit —
/// which means it is invisible to an opcode-keyed arm set, and an object built from the arm set
/// alone would be missing the A4W4 body it needs. This is the same reason `Arm` carries `hd`.
/// The FOUR PRECISION AXES, carried separately.
///
/// The existing booleans (`fp8_weights`, `fp8_kv`, `w8a8`, `a4w4`, …) each answer a yes/no about
/// one arm, and a consumer that wants "what precision is this packet?" has to reconstruct the axes
/// from them — which is a JUDGEMENT, made independently in every consumer, and therefore made
/// differently. `scripts/gfx950_objects.py` carries exactly such a FEATURE→AXIS map by hand.
///
/// These four fields make it a lookup instead. Each is derived from the instruction stream like
/// everything else here, and each names ONE axis:
///
///   * `weight_enc` — what the projection weights are stored as;
///   * `act_enc`    — what the activations are fed to the matmuls as. This is the axis that had no
///     flag and was decided by PHASE, which is how a w8a16 packet reached a w8a8-only object;
///   * `kv_enc`     — the KV cache dtype, independent of the weight axis;
///   * `expert_enc` — the MoE expert weights, which the MLA family carries as a runtime field and
///     which therefore cannot be read off an opcode name at all.
fn precision_axes(
    f: &mut Map<String, Value>,
    s: &Shapes,
    union: &BTreeSet<Arm>,
) -> Map<String, Value> {
    let has = |n: &str| union.iter().any(|a| a.op == n);
    let on = |k: &str| f.get(k).and_then(Value::as_bool).unwrap_or(false);

    // WEIGHT. mxfp4 projections > fp8 (either flavour) > bf16.
    let weight = if s.mxfp4_proj {
        "mxfp4"
    } else if on("fp8_weights") {
        "fp8"
    } else {
        "bf16"
    };
    // ACTIVATION. `QuantFp8` exists only on the w8a8 path — it IS the activation quant, so its
    // presence is the axis, not an inference from a flag. Everything else feeds bf16 activations,
    // including w4a16 and w8a16: narrow weights, wide activations.
    let act = if has("QuantFp8") { "fp8" } else { "bf16" };
    // KV.
    let kv = if on("fp8_kv") { "fp8" } else { "bf16" };
    // EXPERTS. `moe_enc` is the runtime encoding field; absent means there are no expert ops.
    let expert = match (s.moe_enc.len(), s.moe_enc.iter().next()) {
        (0, _) => "none",
        (1, Some(0)) => "bf16",
        (1, Some(1)) => "fp8blk",
        (1, Some(2)) => "mxfp4",
        // More than one encoding on the expert ops IS the mixed case; name it rather than pick.
        _ => "mixed",
    };
    let mut ax = Map::new();
    ax.insert("weight_enc".into(), json!(weight));
    ax.insert("act_enc".into(), json!(act));
    ax.insert("kv_enc".into(), json!(kv));
    ax.insert("expert_enc".into(), json!(expert));
    ax
}

fn encoding_features(f: &mut Map<String, Value>, s: &Shapes) {
    f.insert("a4w4".into(), json!(s.moe_enc.contains(&2)));
    // Correct `fp8_weights` for the encoding field. The `*Fp8Blk` opcodes are the carrier for BOTH
    // block-fp8 and MXFP4 experts, so their presence alone says nothing; what decides it is the
    // encoding those instructions actually carry. Ops that are unconditionally block-fp8 —
    // `GemvFp8Blk` and `DenseGluFp8Blk` have no encoding field — still count on their own.
    let expert_fp8 = s.moe_enc.contains(&1);
    let hard_fp8 = |k: &str| union_has(s, k);
    if f.get("fp8_weights")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && !expert_fp8
    {
        let real = FP8_WEIGHT_OPS.iter().any(|k| hard_fp8(k));
        f.insert("fp8_weights".into(), json!(real));
    }
    // w4a16 MXFP4 on the plain [N,K] projections — a different question from the experts' A4W4,
    // and both have to be true for a packet to be all-MXFP4.
    f.insert("mxfp4_weights".into(), json!(s.mxfp4_proj));
    // A packet is meant to be ALL of one encoding. Two on the grouped ops means a mixed run, which
    // is a thing to see in the manifest rather than to discover in a benchmark number.
    f.insert("moe_enc_mixed".into(), json!(s.moe_enc.len() > 1));
    // Activation-side prefill arms — instruction FIELDS, so they must be surfaced here for the
    // object `requires` derivation (an old object silently heap-overruns on a part16 packet).
    f.insert("moe_pf_part16".into(), json!(s.moe_pf_part16));
    f.insert("moe_pf_atomic".into(), json!(s.moe_pf_atomic));
    f.insert("moe_pf_det".into(), json!(s.moe_pf_det));
    f.insert("moe_pf_a8".into(), json!(s.moe_pf_a8));
    f.insert("xr_combine_fold".into(), json!(s.xr_combine_fold));
    f.insert("kda_fb_fold".into(), json!(s.kda_fb_fold));
    f.insert("kda_decode_fused_arm".into(), json!(s.kda_decode_fused_arm));
    // L8 is packet-inert (loads only), so it is an EMIT setting surfaced here for the paired
    // `plow_config.h`, not an instruction signature.
    f.insert(
        "gemv_prefetch".into(),
        json!(crate::emit_config::active().gemv_prefetch),
    );
    f.insert("moe_prefill_ep".into(), json!(!s.moe_prefill_ep.is_empty()));
    f.insert("quant_glu_fold".into(), json!(s.quant_glu_fold));
    f.insert("mla_pf_ns".into(), json!(s.mla_pf_ns));
    f.insert("glm_ofold".into(), json!(s.glm_ofold));
    f.insert("glm_fuse_rope".into(), json!(s.glm_fuse_rope));
    f.insert("glm_fuse_qnorm".into(), json!(s.glm_fuse_qnorm));
}

/// Performance constants derived by RULE from the shapes, never hardcoded. Both
/// rules below are measured, and both correspond to a failure that has happened:
///
/// * `gv_mm_max = next_pow2(decode_batch)` — `op_gemm.cuh`'s `GV_MM_MAX=16` arm
///   is for B>=16 only. An asset built at 16 and served at B=8 measured −19.4%
///   at 131k and −33.8% at 1k (perf-data/px10-batched-decode.md).
/// * `gf_full = gqa`, but ONLY when the full-attention layers have
///   `kv_heads == 1` — with one KV head the whole GQA group shares a K/V stream
///   and fusing the group re-reads it once instead of `gqa` times. Measured
///   1.48x (perf-data/px11-flash-decode.md). `GF_FULL` must also divide `gqa`
///   or the interpreter traps (`interp_sm120.cu`: `if ((gqa % GF_FULL) != 0)
///   __trap()`), which the `1|2|4|8` clamp below keeps true.
fn tuning(s: &Shapes) -> Map<String, Value> {
    let mut t = Map::new();
    t.insert("gv_mm_max".into(), json!(next_pow2(s.decode_batch.max(1))));
    // TILE PROVENANCE. Written because its absence made a real regression unauditable: for
    // several days every AMD compile selected GEMM tiles from the ANALYTICAL MODEL (both tuning
    // cells were wholly stale against the current build digest) and nothing in the emitted
    // artifact said so -- `pick_tile` reports tier `portable` when it falls back, which is
    // exactly what it reports when no campaign has ever run. A blob has to be able to answer
    // "were my tiles measured?" long after the tree that built it moved on.
    crate::tune_demand::report();
    let (hits, lookups) = crate::tune_demand::tally();
    if lookups > 0 {
        t.insert("tile_lookups".into(), json!(lookups));
        t.insert("tile_measured".into(), json!(hits));
        t.insert(
            "tile_source".into(),
            json!(if hits == 0 {
                "analytical"
            } else if hits == lookups {
                "measured"
            } else {
                "mixed"
            }),
        );
    }
    if s.full_kv_heads == 1 && s.gqa > 0 {
        // The template is instantiated at 1|2|4|8 only.
        let gf = next_pow2(s.gqa).min(8);
        let gf = if s.gqa % gf == 0 { gf } else { 1 };
        t.insert("gf_full".into(), json!(gf));
    }
    t
}

/// Render the neutral facts into ONE toolchain's flags. This is the only place
/// in the manifest pipeline that knows `-D` spellings exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `nvcc` → `.cubin` (sm_120a, sm_90a).
    Nvcc,
}

/// `requires` = CORRECTNESS. A mismatch traps today: the interpreter's dispatch
/// `switch` has a `default: __trap()`, so a packet carrying an opcode the object
/// was not built with is a `CUDA_ERROR_LAUNCH_FAILED` at first launch that reads
/// like a driver bug.
///
/// `recommends` = PERFORMANCE. Wrong here costs throughput, not correctness.
fn backend_nvcc(f: &Map<String, Value>, t: &Map<String, Value>, s: &Shapes) -> Value {
    let on = |k: &str| f.get(k).and_then(Value::as_bool).unwrap_or(false);
    // PLOW_NV_GEMMA=1 only when the packet uses head dims > 128 (Gemma-family
    // hd256/512). A Qwen-only (hd=128) packet must NOT carry this flag — the
    // Gemma build drops the hd=128 arm entirely.
    let mut req = Vec::new();
    if s.hd.iter().any(|&h| h > 128) {
        req.push("PLOW_NV_GEMMA=1".to_string());
    }
    if on("qwen_gdn") {
        req.push("PLOW_NV_QWEN_GDN=1".into());
        req.push("PLOW_NV_FA_GF=2".into());
    }
    if on("w8a8") {
        req.push("PLOW_NV_W8A8=1".into());
        if on("qwen_gdn") {
            req.push("PLOW_NV_FP8_M1=1".into());
            req.push("PLOW_NV_QUANT_FP8_VLLM=1".into());
        }
    }
    if on("fp8_kv") {
        req.push("PLOW_FP8_KV=1".into());
    }
    if on("prefill") {
        req.push("PLOW_NV_PREFILL=1".into());
        if s.prefill_gemv {
            req.push("PLOW_NV_PF_GEMV_HEAD=1".into());
        }
    }
    let mut rec = Vec::new();
    if let Some(v) = t.get("gv_mm_max").and_then(Value::as_u64) {
        rec.push(format!("GV_MM_MAX={v}"));
    }
    if let Some(v) = t.get("gf_full").and_then(Value::as_u64) {
        rec.push(format!("PLOW_NV_FA_GF_FULL={v}"));
    }
    json!({ "requires": req, "recommends": rec })
}

/// The `backends` block: one entry per backend that could serve this packet.
///
/// The AMD entry is keyed by the TARGET arch, not by the literal `gfx950`. A manifest that says
/// `"arch": "gfx942"` beside a `"gfx950"` backend describes an object nobody should build: a
/// consumer following the key compiles for the wrong ISA, and one following `arch` gets a define
/// set tuned for 160 KiB of LDS it does not have. A non-AMD compile keeps the `gfx950` key so the
/// manifest still answers "what would an AMD object need?" for the target this tree has always
/// described, and the NV-side artifacts are unchanged.
fn backends(
    arch: &str,
    f: &Map<String, Value>,
    s: &Shapes,
    union: &BTreeSet<Arm>,
    t: &Map<String, Value>,
) -> Value {
    let amd_key = if arch.starts_with("gfx") {
        arch
    } else {
        "gfx950"
    };
    let gf_full = t.get("gf_full").and_then(Value::as_u64);
    json!({
        "nvcc": backend_nvcc(f, t, s),
        amd_key: backend_amd(amd_key, f, s, union, gf_full),
    })
}

/// Render the neutral facts into the AMD (`hipcc` → `.hsaco`) define set for `arch`.
///
/// `backend_nvcc` was the only renderer, so an AMD build had to be derived by hand from the
/// features — and `scripts/gfx950_objects.py` grew a FEATURE→AXIS map to do exactly that. This is
/// that map, in the file that owns the facts.
///
/// TWO LISTS, and the split is the same one `backend_nvcc` makes: `requires` is CORRECTNESS —
/// leave one out and an opcode has no arm. On AMD that is worse than on NVIDIA, because the
/// dispatch `default:` does not `__trap()`, it writes NOTHING: the packet runs, the buffer keeps
/// whatever was in it, and the failure surfaces as an accuracy bug. Three separate instances of
/// exactly that shape were found in one week (`GEMM_SMALL` missing from the fp8 prefill object,
/// the flash object chosen without following the KV axis, and the KV axis silently dragging the
/// weight axis), which is why this belongs in the manifest rather than in a script.
///
/// NOT rendered here: which OBJECT to build. That is a function of the cmake table
/// (`runtime/CMakeLists.txt`), which pairs define-sets with kernel symbols and register budgets,
/// and restating it here is precisely the drift this file exists to prevent —
/// `scripts/gfx950_objects.py` PARSES that table, which is the right relationship. This renders the
/// defines a covering object must have been built with; matching them to a row is the script's job.
///
/// ARCH-KEYED, not gfx950-only. The two CDNA levels do not take the same define set: CDNA3 has
/// 64 KiB of workgroup LDS against CDNA4's 160 KiB, so gfx950's default GEMM stage arena does not
/// fit and the object silently gets a different tile. Those tile/stage defines therefore belong in
/// `requires` on gfx942 for the same reason every other entry here does — leave them out and the
/// build is wrong rather than merely slow. The authority for the set is
/// `kernelcaps::targets::prefill_recipe`, which is what actually builds the object; this renders
/// the same facts into the manifest so a consumer reading the artifact and a consumer running the
/// recipe cannot disagree.
fn backend_amd(
    arch: &str,
    f: &Map<String, Value>,
    s: &Shapes,
    union: &BTreeSet<Arm>,
    t_gf_full: Option<u64>,
) -> Value {
    let on = |k: &str| f.get(k).and_then(Value::as_bool).unwrap_or(false);
    let has = |n: &str| union.iter().any(|a| a.op == n);

    let mut req: Vec<String> = Vec::new();
    // CDNA3 LDS budget. `GM_DBUF=1` single-buffers the GEMM stage, which is what makes the
    // 192x256 tile fit 64 KiB at eight waves; on CDNA4 the double-buffered default fits and is
    // faster, so this is genuinely arch-conditional and not a tuning preference.
    //
    // Spelled from `hwspec`'s per-arch geometry rather than as literals: this manifest is what a
    // consumer CHECKS an object against, so a stale copy here reports a mismatch on a correctly
    // built object (or, worse, passes a wrongly built one).
    if arch == "gfx942" && !s.prefill_buckets.is_empty() {
        let g = hwspec::IsaLevel::Gfx942
            .geometry()
            .expect("gfx942 geometry");
        req.push("PLOW_WG_WAVES=8".into());
        req.push(format!("GM_DBUF={}", g.gemm_stage_buffers));
        req.push(format!("GM_BM={}", g.gemm_tile.bm));
        req.push(format!("GM_BN={}", g.gemm_tile.bn));
    }
    // Phase. A packet with prefill buckets needs the prefill object; the decode object is always
    // needed because every packet has a decode program.
    if !s.prefill_buckets.is_empty() {
        req.push("PLOW_BUCKET_DECODE=0".into());
    }
    // Weight/activation axis. Both fp8 profiles compile under PLOW_FP8; w8a8 additionally needs
    // the activation-quant arm, and emitting w8a16 for this target is refused upstream anyway.
    if on("fp8_weights") {
        req.push("PLOW_FP8=1".into());
    }
    if on("w8a8") {
        req.push("PLOW_W8A8=1".into());
    }
    if on("fp8_kv") {
        req.push("PLOW_FP8_KV=1".into());
    }
    if on("mxfp4_weights") {
        req.push("PLOW_MXFP4=1".into());
    }
    // MLA family. The attention prefill and the MoE prefill are SEPARATE axes: the grouped expert
    // GEMM is a second full MFMA body in a bucket already at the register cliff, so an
    // attention-only object must not carry it.
    if has("FlashMlaPrefill") || has("FlashMlaPrefillFp8") || has("FlashGatherPrefill") {
        req.push("PLOW_MLA_PREFILL=1".into());
    }
    if on("moe_prefill") {
        req.push("PLOW_MOE_PREFILL=1".into());
    }
    if on("a4w4") {
        req.push("PLOW_MOE_PF_A4W4=1".into());
    }
    if has("KdaConvStateStepG") {
        req.push("PLOW_KDA_CONV_STEP_DB=1".into());
    }
    if on("materialized_residual_input") {
        req.push("PLOW_MATERIALIZED_RESIDUAL_INPUT=1".into());
    }
    if on("attnres_decode_mwg") {
        req.push("PLOW_ATTNRES_DECODE_MWG=1".into());
    }
    if has("KdaChunkPrepare") || has("KdaChunkIntra") || has("KdaChunkWu") || has("KdaChunkCarry") {
        req.push("PLOW_KDA_CHUNK=1".into());
    }
    // Sequence-parallel seams: the split collective arms (ops 25/26) and the host's band-view
    // binding. An object without them would run the packet as a silent no-op.
    if has("XReduceScatter") || has("XAllGather") {
        req.push("PLOW_SEQ_PAR_SEAMS=1".into());
    }
    if union.iter().any(|a| {
        matches!(a.op.as_str(), "KdaChunkWu" | "KdaChunkCarry")
            && a.variant.as_deref().is_some_and(|v| v.ends_with("_qpre"))
    }) {
        req.push("PLOW_KDA_CHUNK_QPRE=1".into());
    }
    // Runtime-flag arms (packet i[7] on ops 85/86/87): every object built since the arms landed
    // carries them (unconditional plow_moe_pf_*_arm markers in op_moe.h); an OLDER object would
    // store f32 into a half-sized part buffer / matmul fp8 bytes as bf16 — refuse at load.
    if on("moe_pf_part16") {
        req.push("PLOW_MOE_PF_PART16=1".into());
    }
    if on("moe_pf_a8") {
        req.push("PLOW_MOE_PF_A8=1".into());
    }
    // L4 combine fold: a BUILD axis of the tagged decode object (`#if PLOW_XR_COMBINE_FOLD`).
    if on("xr_combine_fold") {
        req.push("PLOW_XR_COMBINE_FOLD=1".into());
    }
    // L3 f_b fold: a BUILD axis of the decode object (`#if PLOW_KDA_FB_FOLD`).
    if on("kda_fb_fold") {
        req.push("PLOW_KDA_FB_FOLD=1".into());
    }
    // L7 fused decode arm: a BUILD axis of the decode object (`#if PLOW_KDA_DECODE_FUSED_ARM`).
    if on("kda_decode_fused_arm") {
        req.push("PLOW_KDA_DECODE_FUSED_ARM=1".into());
    }
    // The fused 86->87 decomposition. Unlike the two above this is a BUILD axis (the atomic
    // branch is `#if PLOW_MOE_PF_ATOMIC`), so an object may genuinely not have it.
    if on("moe_pf_atomic") {
        req.push("PLOW_MOE_PF_ATOMIC=1".into());
    }
    // The DETERMINISTIC twin — a BUILD axis for the same reason, and mutually exclusive with it.
    if on("moe_pf_det") {
        req.push("PLOW_MOE_PF_DET=1".into());
    }
    // T11 GLU-into-quant fold (QuantFp8 t3/t4/i2). Same class as the two above and the reason
    // this entry exists at all: the fold deletes the `Glu` packet, and the AMD dispatch ignored
    // t3/t4 for its whole life — so the packet quantized an `fu` nothing had written and the
    // model answered fluently and wrongly. Unconditional arm, so the marker is the whole test.
    if on("quant_glu_fold") {
        req.push("PLOW_T11_GLUQUANT=1".into());
    }
    // Dense causal KV-split partial layout (op 51 i[6] low bits): an older object's V2 arm
    // writes nsplit=1 partials while the merge reads ns of them — refuse at load.
    if on("mla_pf_ns") {
        req.push("PLOW_MLA_PF_NS=1".into());
    }
    // The W_ofold fusion (op 51 i[6] bit 8): the FLASH object must carry the ofold-aware V2
    // arm AND the serve must route MLA prefill there (PLOW_MLA_PF_V2=1) — the 8-wave kernel
    // ignores i[6] and the fused GEMM would read unnormalized f32 partials as bf16. plowrt
    // enforces both.
    if on("glm_ofold") {
        req.push("PLOW_GLM_OFOLD=1".into());
    }
    // The DECODE q-rope fold (op 50 t[7]). The arm is a runtime branch inside
    // `d_flash_mla_decode`, so every object built since it landed carries the marker and every
    // older one does not. Getting this wrong is silent in the worst way: an old object stages
    // `t[3]` as if it were already roped, so the query is UNROPED, attention still runs, and the
    // model answers fluently and wrongly. plowrt refuses the pairing at load.
    if on("glm_fuse_rope") {
        req.push("PLOW_GLM_FUSE_ROPE=1".into());
    }
    // The DECODE q-norm fold (op 22 t[7]). Unlike the rope fold this arm is a BUILD AXIS, not an
    // unconditional runtime branch — an object built without `-DPLOW_GLM_FUSE_QNORM=1` has no
    // fold body at all — so the marker is conditional and its absence means the object cannot
    // run the packet, not merely that it predates it. The failure is the same silent class: the
    // raw pre-norm q_a row goes straight into the projection, attention still runs, and the model
    // answers fluently and wrongly.
    if on("glm_fuse_qnorm") {
        req.push("PLOW_GLM_FUSE_QNORM=1".into());
    }
    // KIMI-K3's BLOCK ops. Not a prefill axis and not a precision one — a model axis, and the only
    // arm flag here that both buckets need. Its absence is the most completely silent failure this
    // list can describe: AttnRes replaces the residual ADD twice in every layer, `situ` is the
    // activation on every GLU, and the KDA recurrence is 69 of 93 mixers. An object without
    // PLOW_K3 skips all of it through the non-trapping `default:` and returns fluent output from
    // a model that is missing most of itself.
    if has("AttnRes")
        || has("SituGlu")
        || has("MlaOutGate")
        || has("KdaStateStep")
        || has("KdaStateStepG")
        || has("KdaConvStateStepG")
        || has("KdaConv")
        || has("KdaConv3")
        || has("KdaGatedNorm")
        || has("KdaChunkPrepare")
        || has("KdaChunkIntra")
        || has("KdaChunkWu")
        || has("KdaChunkCarry")
    {
        req.push("PLOW_K3=1".into());
    }
    // A prefill lm_head emitted as GEMV needs `case PLOW_DOP_GEMV` in the prefill object. It is
    // unconditional there today, so this is documentation of a dependency rather than a flag —
    // recorded in `recommends` so a future object that compiles it out is caught by the pairing
    // hash rather than by a silent no-op.
    let mut rec: Vec<String> = Vec::new();
    // FA_GF_FULL is a PAIRED value: the packet's nsplit is derived from it and the kernel decides
    // how many query heads a work item carries from it, so the two MUST agree or they disagree
    // about the same number. `backend_nvcc` renders the sm_120 spelling; without the AMD spelling
    // here, setting the env var moved only the packet half and the gfx950 kernel kept its built-in
    // default. Same value, both backends, so the pair cannot drift.
    if let Some(v) = t_gf_full {
        rec.push(format!("PLOW_FA_GF_FULL={v}"));
    }
    if !s.prefill_buckets.is_empty() && has("Gemv") {
        rec.push("prefill object must dispatch PLOW_DOP_GEMV (lm_head at M=1)".into());
    }
    json!({ "requires": req, "recommends": rec })
}

/// Which program owns each ceiling, and whether one program forces a ceiling the
/// others do not.
///
/// The object's register/smem footprint is the WORST CASE over every arm compiled
/// into it, and today every prefill bucket lives in one object — so the ceiling is
/// guessed, not attributed. This does not measure registers (that needs `ptxas`);
/// it reports the arm sets that DETERMINE them, which is the part nobody can
/// currently state. A program whose arm set is a strict superset of every other
/// program's of the same kind is a SPLIT CANDIDATE: giving it its own object is
/// the same trick the tree already uses at whole-object granularity (decode vs
/// prefill vs seg-GEMM exist precisely so prefill's hungry arms do not stack onto
/// decode's budget).
///
/// Emitting the recommendation is the whole job here. Actually splitting needs
/// `plowrt` to load and select among several prefill modules, which is a separate
/// and larger piece of work.
fn analysis(progs: &[ProgramArms]) -> Value {
    let mut widest: BTreeMap<&str, (u32, usize)> = BTreeMap::new();
    for p in progs {
        let e = widest.entry(p.kind).or_insert((p.t, p.arms.len()));
        if p.arms.len() > e.1 {
            *e = (p.t, p.arms.len());
        }
    }
    let owns: Vec<Value> = widest
        .iter()
        .map(|(k, (t, n))| json!({ "kind": k, "t": t, "arms": n }))
        .collect();

    // A split candidate carries arms no other program of its kind carries.
    let mut split = Vec::new();
    for p in progs {
        let others: BTreeSet<&Arm> = progs
            .iter()
            .filter(|q| {
                q.kind == p.kind
                    && (q.t != p.t
                        || q.seg != p.seg
                        || q.packed_prefill_only != p.packed_prefill_only)
            })
            .flat_map(|q| q.arms.iter())
            .collect();
        let uniq: Vec<String> = p
            .arms
            .iter()
            .filter(|a| !others.contains(a))
            .map(Arm::key)
            .collect();
        if !uniq.is_empty() && progs.iter().filter(|q| q.kind == p.kind).count() > 1 {
            split.push(json!({
                "kind": p.kind, "t": p.t, "segment": p.seg, "exclusive_arms": uniq,
            }));
        }
    }
    json!({
        "ceiling_owner": owns,
        "split_candidates": split,
        "note": "arm-set attribution only; register/smem numbers need `ptxas -v` on the built \
                 object. A per-program object split is deliberately NOT implemented here — it \
                 needs plowrt to load and select among several prefill modules.",
    })
}

/// Build the manifest for an emitted [`Model`].
///
/// `arch` is the target triple-ish name (`"sm_120a"`), carried through so the
/// backend that renders flags knows what it is rendering for; it is metadata,
/// not something this module interprets.
pub fn build(m: &Model, arch: &str, lean: &crate::LeanReport) -> Value {
    let mut v = build_inner(m, arch, lean);
    // Stamped last: it is a hash OF the manifest's compiled-set fields, so it
    // cannot be one of them.
    let h = pairing_hash(&v);
    v["pairing"] = json!({
        "hash": format!("0x{h:016x}"),
        "algo": "fnv1a64 over `union`, `objects`, then `tuning`",
        "note": "A cubin built from this manifest stamps this value as \
                 plow_packet_hash_{lo,hi}; plowrt refuses a module whose stamp \
                 disagrees. A GENERAL object (every arm compiled) carries no stamp \
                 and pairs with any packet.",
    });
    v
}

/// The `lean` block: WAS THIS BLOB VERIFIED, and if not, why not.
///
/// The gate is on by default and DEGRADES — no `plow_verify` on the machine
/// means warn-and-skip, not a failed compile. That is the right behaviour and
/// it creates the exact hazard this repo keeps getting caught by: a skipped
/// gate and a passing gate look identical from the outside. `tuning.tier`
/// already burned us this way (`portable` meant both "analytical model chosen"
/// and "nothing was ever measured"), and the GLM prefill numbers taken on top
/// of it were meaningless for weeks.
///
/// So a downstream consumer — a benchmark harness, a reviewer, a bisect — must
/// be able to ask the ARTIFACT, not a build log it no longer has.
///
/// `verified: false` NEVER means "rejected": a rejection panics before the blob
/// is written, so no manifest can describe a rejected program.
fn lean_block(lean: &crate::LeanReport) -> Value {
    json!({
        "verified": lean.verified,
        "oracle": lean.oracle,
        "reason": lean.reason,
        "note": "`verified` = a Lean ordering certificate (plow_verify checkpoint D) was \
                 obtained for EVERY program in this blob; `oracle` = the Lean decode \
                 lower-bound query ran. false means NOT CHECKED (see `reason`), never \
                 `checked and rejected` — a rejection aborts emission, so a blob with a \
                 rejected program never reaches disk and has no manifest.",
    })
}

fn object_inventory(progs: &[ProgramArms], arch: &str) -> Value {
    let phase = |kind: &str| -> BTreeSet<Arm> {
        progs
            .iter()
            .filter(|p| p.kind == kind)
            .flat_map(|p| p.arms.iter().cloned())
            .collect()
    };
    let prefill = phase("prefill");
    let decode_mla_segment = |p: &&ProgramArms| {
        p.kind == "decode"
            && p.seg.is_some()
            && p.insts == 2
            && p.arms.len() == 2
            && p.arms.iter().any(|a| a.op == "FlashMlaDecode")
            && p.arms.iter().any(|a| a.op == "MlaMergeFold")
            && p.arms
                .iter()
                .all(|a| a.op == "FlashMlaDecode" || a.op == "MlaMergeFold")
    };
    let decode_mla: BTreeSet<Arm> = progs
        .iter()
        .filter(decode_mla_segment)
        .flat_map(|p| p.arms.iter().cloned())
        .collect();
    let decode: BTreeSet<Arm> = progs
        .iter()
        .filter(|p| p.kind == "decode" && !decode_mla_segment(p))
        .flat_map(|p| p.arms.iter().cloned())
        .collect();
    let flash_family = |arm: &&Arm| arm.op.starts_with("Flash") || arm.op == "MlaMergeFold";
    let keys = |arms: &BTreeSet<Arm>| arms.iter().map(Arm::key).collect::<Vec<_>>();
    let families = |arms: &BTreeSet<Arm>| {
        arms.iter()
            .map(|a| opcode_family(&a.op))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    };
    let excluded = |arms: &BTreeSet<Arm>| {
        DevOp::ALL
            .iter()
            .map(|op| op_name(*op))
            .filter(|name| !arms.iter().any(|a| a.op == *name))
            .collect::<Vec<_>>()
    };
    let resource_contract = |flash: bool| {
        if arch.starts_with("gfx") {
            json!({
                "max_total_registers": if flash { 512 } else { 256 },
                "min_occupancy_waves_per_simd": if flash { 1 } else { 2 },
                "wavefront_size": 64,
                "policy": "refuse",
            })
        } else {
            Value::Null
        }
    };
    let flash: BTreeSet<Arm> = prefill.iter().filter(flash_family).cloned().collect();
    let packed_kda_names = ["KdaStateStep", "KdaConv3", "KdaStateStepG"];
    let packed_kda: BTreeSet<Arm> = prefill
        .iter()
        .filter(|a| packed_kda_names.contains(&a.op.as_str()))
        .cloned()
        .collect();
    let fused: BTreeSet<Arm> = decode
        .iter()
        .filter(|a| a.op == "KdaDecodeFused")
        .cloned()
        .collect();
    let singleton_arm = |name: &str| -> BTreeSet<Arm> {
        progs
            .iter()
            .filter(|p| p.kind == "prefill" && p.seg.is_some() && p.insts == 1)
            .filter_map(|p| {
                (p.arms.len() == 1)
                    .then(|| p.arms.iter().next().unwrap())
                    .filter(|a| a.op == name && a.variant.as_deref() == Some("d128_qpre"))
                    .cloned()
            })
            .collect()
    };
    let key_factor_wu = singleton_arm("KdaChunkWu");
    let key_factor_carry = singleton_arm("KdaChunkCarry");
    let key_factor_pair = !key_factor_wu.is_empty() && !key_factor_carry.is_empty();
    json!({
        "ordinary": {
            "prefill": {
                "arms": keys(&prefill),
                "families": families(&prefill),
                "excluded_opcodes": excluded(&prefill),
                "resource_contract": resource_contract(false),
                "inventory_prune_capable": true,
            },
            "decode": {
                "arms": keys(&decode),
                "families": families(&decode),
                "excluded_opcodes": excluded(&decode),
                "resource_contract": resource_contract(false),
                "inventory_prune_capable": true,
            },
            "decode_mla": {
                "required": !decode_mla.is_empty(),
                "arms": keys(&decode_mla),
                "families": families(&decode_mla),
                "resource_contract": resource_contract(false),
            },
            "flash": {
                "arms": keys(&flash),
                "families": families(&flash),
                "resource_contract": resource_contract(true),
            },
        },
        "lean": {
            "packed_kda_prefill": { "required": !packed_kda.is_empty(), "arms": keys(&packed_kda) },
            "kda_decode_fused": { "required": !fused.is_empty(), "arms": keys(&fused) },
            "kda_key_factor_pair": {
                "required": key_factor_pair,
                "wu_arms": keys(&key_factor_wu),
                "carry_arms": keys(&key_factor_carry),
            },
        },
    })
}

/// Stable, model-neutral family labels for object partitioning and reports.
/// They are derived only from emitted opcodes. They do not decide correctness;
/// the exact arm list above remains the executable inventory.
fn opcode_family(op: &str) -> &'static str {
    if op.starts_with("Kda") {
        "kda"
    } else if op.starts_with("Moe") {
        "moe"
    } else if op.starts_with("FlashMla")
        || op.starts_with("FlashGather")
        || matches!(op, "MlaMergeFold" | "MlaOutGate" | "MlaMaterializePack")
    {
        "mla"
    } else if op.starts_with("Flash") || op == "AttnSelect" {
        "attention"
    } else if op.starts_with("Gemm") || op.starts_with("Gemv") {
        "linear"
    } else if op.contains("Norm") || matches!(op, "Residual" | "AttnRes") {
        "norm_residual"
    } else if matches!(
        op,
        "XReduce" | "XReduceTwoShot" | "XReduceScatter" | "XAllGather"
    ) {
        "collective"
    } else {
        "elementwise"
    }
}

/// Ordered, graph-derived launch chains.  This is deliberately descriptive:
/// the object builder consumes each segment's exact arm inventory and must
/// attach measured code-object resources before the runtime may select it.
/// Keeping the boundary in the compiler manifest prevents a model-name table
/// from becoming a second, drifting scheduler.
fn dispatch_chains(progs: &[ProgramArms], arch: &str) -> Vec<Value> {
    let resource_contract = |flash: bool| {
        if arch.starts_with("gfx") {
            json!({
                "max_total_registers": if flash { 512 } else { 256 },
                "min_occupancy_waves_per_simd": if flash { 1 } else { 2 },
                "wavefront_size": 64,
                "max_private_segment_bytes_delta": 0,
                "max_vgpr_spill_delta": 0,
                "max_sgpr_spill_delta": 0,
                "policy": "refuse",
            })
        } else {
            Value::Null
        }
    };
    let mut out = Vec::new();
    for program in progs.iter().map(|p| p.program).collect::<BTreeSet<_>>() {
        let rows: Vec<&ProgramArms> = progs.iter().filter(|p| p.program == program).collect();
        let Some(first) = rows.first() else { continue };
        let segment_values: Vec<Value> = rows
            .iter()
            .map(|p| {
                let families: BTreeSet<_> = p.arms.iter().map(|a| opcode_family(&a.op)).collect();
                let flash = p
                    .arms
                    .iter()
                    .any(|a| a.op.starts_with("Flash") || a.op == "MlaMergeFold");
                json!({
                    "segment": p.seg.unwrap_or(0),
                    "insts": p.insts,
                    "arms": p.arms.iter().map(Arm::key).collect::<Vec<_>>(),
                    "families": families,
                    "object_class": if flash { "flash" } else { "ordinary" },
                    "resource_contract": resource_contract(flash),
                })
            })
            .collect();
        let mut phases = Vec::new();
        let mut lo = 0usize;
        while lo < rows.len() {
            let families: BTreeSet<_> =
                rows[lo].arms.iter().map(|a| opcode_family(&a.op)).collect();
            let flash = rows[lo]
                .arms
                .iter()
                .any(|a| a.op.starts_with("Flash") || a.op == "MlaMergeFold");
            let mut hi = lo + 1;
            while hi < rows.len() {
                let next_families: BTreeSet<_> =
                    rows[hi].arms.iter().map(|a| opcode_family(&a.op)).collect();
                let next_flash = rows[hi]
                    .arms
                    .iter()
                    .any(|a| a.op.starts_with("Flash") || a.op == "MlaMergeFold");
                if next_families != families || next_flash != flash {
                    break;
                }
                hi += 1;
            }
            let arms: BTreeSet<_> = rows[lo..hi]
                .iter()
                .flat_map(|p| p.arms.iter().map(Arm::key))
                .collect();
            phases.push(json!({
                "first_segment": rows[lo].seg.unwrap_or(0),
                "last_segment": rows[hi - 1].seg.unwrap_or(0),
                "segments": hi - lo,
                "arms": arms,
                "families": families,
                "object_class": if flash { "flash" } else { "ordinary" },
                "resource_contract": resource_contract(flash),
            }));
            lo = hi;
        }
        out.push(json!({
            "program": program,
            "kind": first.kind,
            "topology": if first.packed_prefill_only { "packed" } else { "ordinary" },
            if first.kind == "prefill" { "bucket" } else { "batch" }: first.t,
            "segments": segment_values,
            "phases": phases,
            "aql_replay": {
                "packets": rows.len(),
                "ordering": "barrier_agent_scope",
                "rank_commit": "prepare_all_ranks_then_ring_all",
                "host_drains": 1,
            },
        }));
    }
    out
}

fn build_inner(m: &Model, arch: &str, lean: &crate::LeanReport) -> Value {
    let progs = program_arms(m);
    let union: BTreeSet<Arm> = progs.iter().flat_map(|p| p.arms.iter().cloned()).collect();
    let mut objects = object_inventory(&progs, arch);
    let kda_intra_wave_items_required = m.progs.iter().any(|p| {
        p.stream.iter().any(|e| {
            e.flags & packet::dev::SE_KDA_INTRA_WAVE_ITEMS != 0
                && p.insts.get(e.inst as usize).map(|d| d.op) == Some(DevOp::KdaChunkIntra as u16)
        })
    });
    objects["lean"]["kda_intra_wave_items"] = json!({
        "required": kda_intra_wave_items_required,
    });
    let attn_res_f32mix_required = m
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(packet::devbuild::lean_attn_res_f32mix_inst);
    objects["lean"]["attn_res_f32mix"] = json!({
        "required": attn_res_f32mix_required,
    });
    let kda_carry_regstate_required = m.progs.iter().any(|p| {
        p.stream.iter().any(|e| {
            e.flags & packet::dev::SE_KDA_CARRY_REGSTATE != 0
                && p.insts.get(e.inst as usize).map(|d| d.op) == Some(DevOp::KdaChunkCarry as u16)
        })
    });
    objects["lean"]["kda_carry_regstate"] = json!({
        "required": kda_carry_regstate_required,
    });
    let marked_wu = |keys: bool| {
        m.progs.iter().any(|p| {
            p.stream.iter().any(|e| {
                e.flags & packet::dev::SE_KDA_WU_LEAN != 0
                    && p.insts
                        .get(e.inst as usize)
                        .is_some_and(|d| d.op == DevOp::KdaChunkWu as u16 && (d.i[5] == 1) == keys)
            })
        })
    };
    objects["lean"]["kda_wu_lean"] = json!({ "required": marked_wu(false) });
    objects["lean"]["kda_carry_keyfeed"] = json!({ "required": marked_wu(true) });
    let s = shapes(m);
    let mut ep_tables = BTreeSet::new();
    let ep_extra_resident_bytes_per_rank = m
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .filter(|d| d.op == DevOp::MoeGroupGluPf as u16 && d.i[6] > 1 && ep_tables.insert(d.t[2]))
        .map(|d| {
            let degree = u64::from(d.i[6]);
            let local_experts = u64::from(d.i[2]).div_ceil(degree);
            let h = u64::from(d.i[1]);
            let i = u64::from(d.i[0]);
            let matrix_payload = h * i / 2;
            let matrix_scales = h * (i / 32);
            let has_moe2 = m
                .tensors
                .get(d.t[2] as usize)
                .and_then(|t| t.name.strip_suffix("expert_weight_table_ep"))
                .is_some_and(|pfx| {
                    m.tensors
                        .iter()
                        .any(|t| t.name == format!("{pfx}expert_weight_table_moe2_ep"))
                });
            let moe2 = if has_moe2 {
                matrix_payload + h.div_ceil(256) * 256 * ((i / 32).div_ceil(8) * 8)
            } else {
                0
            };
            local_experts * (3 * (matrix_payload + matrix_scales) + moe2)
        })
        .sum::<u64>();
    objects["lean"]["moe_prefill_ep"] = json!({
        "required": !s.moe_prefill_ep.is_empty(),
        "boundaries": s.moe_prefill_ep.iter().map(|(degree, experts, full_i)| json!({
            "degree": degree,
            "experts": experts,
            "full_intermediate_width": full_i,
            "ownership": "balanced_contiguous_whole_experts",
        })).collect::<Vec<_>>(),
        "objects": ["moe_ep_align", "moe_stage1_mxfp4", "moe_ep_stage2", "moe_ep_combine"],
        "additional_resident_bytes_per_rank": ep_extra_resident_bytes_per_rank,
        "capacity_ack_env": "PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES",
        "resource_contract": {
            "wavefront_size": 64,
            "private_segment_bytes": 0,
            "policy": "refuse",
        },
    });
    // Opt-in lean MoE body variants: keys exist only when requested so the default
    // manifest, pairing hash, and config header are unchanged.
    let cfg = crate::emit_config::active();
    if cfg.moe_stage1_body {
        objects["lean"]["moe_stage1_body"] =
            json!({ "required": true, "define": "PLOW_MOE1_BODY" });
    }
    if cfg.moe_stage2_body {
        objects["lean"]["moe_stage2_body"] =
            json!({ "required": true, "define": "PLOW_MOE2_BODY" });
    }
    let mut f = features(&union);
    let materialized_residual_input = m.progs.iter().flat_map(|p| &p.insts).any(|inst| {
        inst.op == DevOp::AttnRes as u16
            && (inst.t[6] != packet::TENSOR_NONE
                || inst.t[7] != packet::TENSOR_NONE
                || inst.i[5] != packet::dev::TENSOR_NONE_I)
    });
    f.insert(
        "materialized_residual_input".into(),
        json!(materialized_residual_input),
    );
    let attnres_decode_mwg = m
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(|inst| inst.op == DevOp::AttnRes as u16 && inst.i[6] != 0);
    f.insert("attnres_decode_mwg".into(), json!(attnres_decode_mwg));
    encoding_features(&mut f, &s);
    f.insert("linear_bias".into(), json!(s.linear_bias));
    f.insert("rope_half_hd64".into(), json!(s.rope_half_hd64));
    f.insert("attention_sinks".into(), json!(s.attention_sinks));
    let axes = precision_axes(&mut f, &s, &union);
    let t = tuning(&s);
    let attention: Vec<Value> = crate::attention_decisions()
        .into_iter()
        .map(|d| {
            json!({
                "cell": {
                    "hardware": d.hardware,
                    "n_cu": d.n_cu,
                    "decode_rung": d.decode_rung,
                    "kv_bucket": d.kv_bucket,
                    "shape": d.shape,
                },
                "compiled": {
                    "algorithms": ["split_reduce"],
                    "max_nsplit": d.compiled_max_nsplit,
                    "persistent": d.compiled_persistent,
                },
                "qualified": d.selected_source == "qualified",
                "selected": {
                    "algorithm": d.selected_algorithm,
                    "nsplit": d.selected_nsplit,
                    "source": d.selected_source,
                },
            })
        })
        .collect();

    let opcodes: BTreeSet<&str> = union.iter().map(|a| a.op.as_str()).collect();
    let programs: Vec<Value> = progs
        .iter()
        .map(|p| {
            let mut o = Map::new();
            o.insert("kind".into(), json!(p.kind));
            o.insert(
                "topology".into(),
                json!(if p.packed_prefill_only {
                    "packed"
                } else {
                    "ordinary"
                }),
            );
            // `bucket` for prefill (chunk rows), `batch` for decode — same field,
            // different meaning, so name it for what it is on each side.
            o.insert(
                if p.kind == "prefill" {
                    "bucket".into()
                } else {
                    "batch".to_string()
                },
                json!(p.t),
            );
            o.insert("segment".into(), json!(p.seg));
            o.insert("insts".into(), json!(p.insts));
            o.insert(
                "arms".into(),
                json!(p.arms.iter().map(Arm::key).collect::<Vec<_>>()),
            );
            Value::Object(o)
        })
        .collect();

    let mut kv_dtype = Map::new();
    for (hd, d) in &s.kv_dtype {
        kv_dtype.insert(format!("hd{hd}"), json!(d));
    }

    json!({
        "schema": 1,
        "arch": arch,
        "n_cu": m.n_cu,
        "input_contract": {
            "kind": "token_ids",
            "modalities": ["text"],
            "vision": false,
        },
        "opcodes": opcodes,
        "shapes": {
            "hd": s.hd,
            "kv_heads": s.kv_heads,
            "gqa": s.gqa,
            "decode_batch": s.decode_batch,
            "kv_dtype": kv_dtype,
            "max_chunk": s.max_chunk,
            "prefill_buckets": s.prefill_buckets,
            "moe_enc": s.moe_enc,
            "moe_prefill_ep": s.moe_prefill_ep,
        },
        "features": f,
        // The four precision axes, so "what precision is this packet?" is a lookup rather than a
        // judgement reconstructed from the feature booleans by every consumer separately.
        "precision": axes,
        "tuning": t,
        "attention_policy": {
            "entries": attention,
            "runtime_kv_reselection": false,
            "note": "selection is exact-cell and offline; runtime KV-bucket reselection remains disabled until calibrated records and packet immediate patching are available",
        },
        // Sits beside `tuning` on purpose: both answer "how much do I trust this
        // artifact?", and both have a value that means "not established".
        "lean": lean_block(lean),
        "programs": programs,
        "dispatch_chains": dispatch_chains(&progs, arch),
        "objects": objects,
        // What a specialised object must compile: the union over every program
        // and segment. Anything narrower and some bucket hits `default: __trap()`.
        "union": union.iter().map(Arm::key).collect::<Vec<_>>(),
        "analysis": analysis(&progs),
        "backends": backends(arch, &f, &s, &union, &t),
    })
}

/// The pairing hash: identifies the (packet, object-set) pair.
///
/// A specialised object is no longer interchangeable — it carries exactly the
/// arms one packet needs — so pairing it with a different packet must be
/// IMPOSSIBLE, not merely discouraged. Without this check, specialisation turns
/// today's loud first-launch `__trap()` into something strictly worse: an object
/// that is missing the arm some later bucket needs and traps mid-serve.
///
/// Hashed over the `union`, per-object inventory, and tuning constants, i.e. exactly what the
/// generated `plow_config.h` compiles — NOT the whole manifest, so a cosmetic
/// field (a comment, a reordered analysis note) does not invalidate a good pair.
/// FNV-1a 64: the runtime already uses FNV for `gpu_fingerprint`, and this is an
/// identity check, not a security boundary.
pub fn pairing_hash(manifest: &Value) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |s: &str| {
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    };
    if let Some(a) = manifest.get("union").and_then(Value::as_array) {
        for v in a {
            feed(v.as_str().unwrap_or(""));
            feed("\x1f");
        }
    }
    feed("\x1e");
    if let Some(objects) = manifest.get("objects") {
        feed(&objects.to_string());
    }
    feed("\x1e");
    if let Some(o) = manifest.get("tuning").and_then(Value::as_object) {
        for (k, v) in o {
            feed(k);
            feed("=");
            feed(&v.to_string());
            feed("\x1f");
        }
    }
    h
}

/// Generate the header a specialised object compiles against.
///
/// Two halves, and both matter:
///  * PRESENCE macros (`PLOW_HAS_FLASH_DECODE_FP8`, `PLOW_HAS_MLA`), so an op arm
///    can be gated on what the packet contains instead of on the hand-maintained
///    `#if` maze;
///  * the SHAPE CONSTANTS the rules produced (`PLOW_GF_FULL`, `GV_MM_MAX`), so the
///    two measured performance rules are applied by construction.
///
/// The existing knobs are NOT replaced — every macro here is emitted `#ifndef`-
/// guarded, so an explicit `-D` on the command line still wins and the A/B
/// controls keep working. The header only supplies values nothing else set.
pub fn write_config_header(path: &std::path::Path, manifest: &Value) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("h.tmp.{}", std::process::id()));
    if let Err(error) = std::fs::write(&tmp, config_header(manifest)) {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

pub fn config_header(manifest: &Value) -> String {
    let mut out = String::new();
    out.push_str(
        "/* GENERATED by devgen::manifest — do not edit.\n\
         *\n\
         * The arm set and shape constants of ONE packet. Every macro is #ifndef-\n\
         * guarded: an explicit -D still wins, so the hand-maintained knobs stay\n\
         * usable as A/B controls and this header only supplies what nothing else\n\
         * set. Pair it with the packet whose PLOW_PACKET_HASH it carries — the\n\
         * loader refuses a mismatch. */\n#pragma once\n\n",
    );
    out.push_str(&format!(
        "#define PLOW_PACKET_HASH 0x{:016x}ull\n\n",
        pairing_hash(manifest)
    ));

    // Presence macros, one per opcode in the union. Named from the `dev_isa.h`
    // spelling so a reader can grep the macro straight to the dispatch case.
    let union: Vec<&str> = manifest
        .get("union")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut ops: BTreeSet<String> = BTreeSet::new();
    for k in &union {
        ops.insert(k.split('/').next().unwrap_or(k).to_string());
    }
    let object_ops = |phase: &str| -> BTreeSet<String> {
        manifest
            .pointer(&format!("/objects/ordinary/{phase}/arms"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|key| key.split('/').next().unwrap_or(key).to_string())
            .collect()
    };
    let prefill_ops = object_ops("prefill");
    let decode_ops = object_ops("decode");
    let flash_ops = object_ops("flash");
    let decode_mla_ops = object_ops("decode_mla");
    let decode_mla_required = manifest
        .pointer("/objects/ordinary/decode_mla/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let kda_intra_wave_items_required = manifest
        .pointer("/objects/lean/kda_intra_wave_items/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let kda_carry_regstate_required = manifest
        .pointer("/objects/lean/kda_carry_regstate/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let kda_wu_lean_required = manifest
        .pointer("/objects/lean/kda_wu_lean/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let kda_carry_keyfeed_required = manifest
        .pointer("/objects/lean/kda_carry_keyfeed/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let moe_prefill_ep_required = manifest
        .pointer("/objects/lean/moe_prefill_ep/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let attn_res_f32mix_required = manifest
        .pointer("/objects/lean/attn_res_f32mix/required")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let kda_chunk_qpre = union
        .iter()
        .any(|arm| arm.starts_with("KdaChunk") && arm.ends_with("_qpre"));
    out.push_str("/* --- packet and per-object opcode inventory --- */\n");
    out.push_str(&format!(
        "#define PLOW_PACKET_HAS_DECODE_MLA_SEGMENTS {}\n",
        if decode_mla_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_INTRA_WAVE_ITEMS {}\n",
        if kda_intra_wave_items_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_ATTN_RES_F32MIX {}\n",
        if attn_res_f32mix_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_CARRY_REGSTATE {}\n",
        if kda_carry_regstate_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_WU_LEAN {}\n",
        if kda_wu_lean_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_CARRY_KEYFEED {}\n",
        if kda_carry_keyfeed_required { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_MOE_PREFILL_EP {}\n",
        if moe_prefill_ep_required { 1 } else { 0 }
    ));
    for (key, macro_name) in [
        ("moe_stage1_body", "PLOW_OBJECT_MOE_STAGE1_BODY"),
        ("moe_stage2_body", "PLOW_OBJECT_MOE_STAGE2_BODY"),
    ] {
        if manifest
            .pointer(&format!("/objects/lean/{key}/required"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            out.push_str(&format!("#define {macro_name} 1\n"));
        }
    }
    for o in DevOp::ALL {
        let name = op_name(*o);
        let present = ops.contains(&name);
        let m = o.c_name().replace("PLOW_DOP_", "PLOW_HAS_");
        let packet_m = o.c_name().replace("PLOW_DOP_", "PLOW_PACKET_HAS_");
        out.push_str(&format!(
            "#define {packet_m} {}\n",
            if present { 1 } else { 0 }
        ));
        out.push_str(&format!(
            "#ifndef {m}\n#if defined(PLOW_BUCKET_FLASH)\n#define {m} {}\n#elif defined(PLOW_BUCKET_DECODE_MLA)\n#define {m} {}\n#elif PLOW_BUCKET_DECODE\n#define {m} {}\n#else\n#define {m} {}\n#endif\n#endif\n",
            if flash_ops.contains(&name) { 1 } else { 0 },
            if decode_mla_ops.contains(&name) { 1 } else { 0 },
            if decode_ops.contains(&name) { 1 } else { 0 },
            if prefill_ops.contains(&name) { 1 } else { 0 },
        ));
    }

    out.push_str(
        "\n/* --- model-neutral family switches derived from object opcode presence --- */\n\
#ifndef PLOW_K3\n\
#define PLOW_K3 (PLOW_HAS_KDA_CONV || PLOW_HAS_KDA_GATE || PLOW_HAS_KDA_STATE_STEP || \\\n PLOW_HAS_KDA_GATED_NORM || PLOW_HAS_ATTN_RES || PLOW_HAS_SITU_GLU || \\\n PLOW_HAS_MLA_OUT_GATE || PLOW_HAS_KDA_CONV3 || PLOW_HAS_KDA_STATE_STEP_G || \\\n PLOW_HAS_KDA_CONV_STATE_STEP_G || PLOW_HAS_KDA_CHUNK_PREPARE || \\\n PLOW_HAS_KDA_CHUNK_INTRA || PLOW_HAS_KDA_CHUNK_WU || PLOW_HAS_KDA_CHUNK_CARRY)\n\
#endif\n\
#ifndef PLOW_KDA_CHUNK\n\
#define PLOW_KDA_CHUNK (PLOW_HAS_KDA_CHUNK_PREPARE || PLOW_HAS_KDA_CHUNK_INTRA || \\\n PLOW_HAS_KDA_CHUNK_WU || PLOW_HAS_KDA_CHUNK_CARRY)\n\
#endif\n\
#ifndef PLOW_KDA_CONV_STEP_DB\n\
#define PLOW_KDA_CONV_STEP_DB PLOW_HAS_KDA_CONV_STATE_STEP_G\n\
#endif\n",
    );
    out.push_str(&format!(
        "#ifndef PLOW_KDA_CHUNK_QPRE\n#define PLOW_KDA_CHUNK_QPRE {}\n#endif\n",
        if kda_chunk_qpre { 1 } else { 0 }
    ));
    let materialized_residual_input = manifest
        .pointer("/features/materialized_residual_input")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#ifndef PLOW_MATERIALIZED_RESIDUAL_INPUT\n#define PLOW_MATERIALIZED_RESIDUAL_INPUT {}\n#endif\n",
        if materialized_residual_input { 1 } else { 0 }
    ));
    let xr_combine_fold = manifest
        .pointer("/features/xr_combine_fold")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_XR_COMBINE_FOLD {0}\n#ifndef PLOW_XR_COMBINE_FOLD\n#define PLOW_XR_COMBINE_FOLD {0}\n#endif\n",
        if xr_combine_fold { 1 } else { 0 }
    ));
    let attnres_decode_mwg = manifest
        .pointer("/features/attnres_decode_mwg")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#ifndef PLOW_ATTNRES_DECODE_MWG\n#define PLOW_ATTNRES_DECODE_MWG {}\n#endif\n",
        if attnres_decode_mwg { 1 } else { 0 }
    ));
    let kda_fb_fold = manifest
        .pointer("/features/kda_fb_fold")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_FB_FOLD {0}\n#ifndef PLOW_KDA_FB_FOLD\n#define PLOW_KDA_FB_FOLD {0}\n#endif\n",
        if kda_fb_fold { 1 } else { 0 }
    ));
    let kda_decode_fused_arm = manifest
        .pointer("/features/kda_decode_fused_arm")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#define PLOW_PACKET_REQUIRES_KDA_DECODE_FUSED_ARM {0}\n#ifndef PLOW_KDA_DECODE_FUSED_ARM\n#define PLOW_KDA_DECODE_FUSED_ARM {0}\n#endif\n",
        if kda_decode_fused_arm { 1 } else { 0 }
    ));
    let gemv_prefetch = manifest
        .pointer("/features/gemv_prefetch")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "#ifndef PLOW_GEMV_PREFETCH\n#define PLOW_GEMV_PREFETCH {}\n#endif\n",
        if gemv_prefetch { 1 } else { 0 }
    ));

    // Head dims the flash family is instantiated at.
    out.push_str("\n/* --- flash head dims present --- */\n");
    for hd in [64u32, 128, 256, 512] {
        let present = union.iter().any(|k| k.ends_with(&format!("/hd{hd}")));
        out.push_str(&format!(
            "#ifndef PLOW_HAS_FLASH_HD{hd}\n#define PLOW_HAS_FLASH_HD{hd} {}\n#endif\n",
            if present { 1 } else { 0 }
        ));
    }
    for (feature, define) in [
        ("linear_bias", "PLOW_PACKET_LINEAR_BIAS"),
        ("rope_half_hd64", "PLOW_PACKET_ROPE_HALF_HD64"),
        ("attention_sinks", "PLOW_PACKET_ATTENTION_SINKS"),
    ] {
        let present = manifest
            .pointer(&format!("/features/{feature}"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        out.push_str(&format!(
            "#define {define} {}\n",
            if present { 1 } else { 0 }
        ));
    }
    out.push_str("\n/* --- decode head-normalization dimensions present --- */\n");
    for hd in [64u32, 128, 256, 512] {
        let present = decode_ops.contains("HeadNormRope")
            && manifest
                .pointer("/objects/ordinary/decode/arms")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .any(|k| k == format!("HeadNormRope/hd{hd}"));
        out.push_str(&format!(
            "#ifndef PLOW_HAS_HEADNORM_HD{hd}\n#define PLOW_HAS_HEADNORM_HD{hd} {}\n#endif\n",
            if present { 1 } else { 0 }
        ));
    }

    out.push_str("\n/* --- rule-derived shape constants --- */\n");
    if let Some(t) = manifest.get("tuning").and_then(Value::as_object) {
        if let Some(v) = t.get("gv_mm_max").and_then(Value::as_u64) {
            out.push_str(&format!(
                "#ifndef GV_MM_MAX\n#define GV_MM_MAX {v}\n#endif\n"
            ));
        }
        if let Some(v) = t.get("gf_full").and_then(Value::as_u64) {
            out.push_str(&format!(
                "#ifndef PLOW_NV_FA_GF_FULL\n#define PLOW_NV_FA_GF_FULL {v}\n#endif\n"
            ));
        }
    }
    if let Some(gqa) = manifest.pointer("/shapes/gqa").and_then(Value::as_u64) {
        out.push_str(&format!("#define PLOW_PACKET_GQA {gqa}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevInst;
    use packet::devbuild::Program;

    /// Shadows [`super::build`]: most assertions here are about arms/shapes and
    /// do not care about the lean block, which has its own tests below.
    fn build(m: &Model, arch: &str) -> Value {
        super::build(m, arch, &crate::LeanReport::skipped("test: gate not run"))
    }

    fn inst(op: DevOp, i: [u32; 8]) -> DevInst {
        DevInst {
            op: op as u16,
            blocks: 1,
            i,
            ..Default::default()
        }
    }

    fn prog(insts: Vec<DevInst>) -> Program {
        Program {
            hier_base: 0,
            n_cu: 4,
            n_counter: 0,
            insts,
            stream: vec![],
            stream_ofs: vec![],
            stream_len: vec![],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        }
    }

    /// Gemma-4-shaped: a prefill bucket + a B=8 decode, sliding hd256 bf16 and
    /// full hd512 fp8 with one KV head.
    fn model() -> Model {
        let pf = prog(vec![
            inst(DevOp::FlashPrefill, [0, 0, 8, 4, 0, 0, 256, 0]),
            inst(DevOp::Gemm, [0; 8]),
        ]);
        let dec = prog(vec![
            // i0=n_batch i1=n_head i2=n_kv_head … i6=hd
            inst(DevOp::FlashDecode, [8, 8, 4, 0, 0, 0, 256, 0]),
            inst(DevOp::FlashDecodeFp8, [8, 8, 1, 0, 0, 0, 512, 0]),
            inst(DevOp::Gemv, [0; 8]),
        ]);
        Model {
            n_cu: 170,
            target: 0,
            tensors: vec![],
            progs: vec![pf, dec],
            kv_row_insts: vec![],
            prog_t: vec![1024, 8],
            gen: vec![],
        }
    }

    /// The manifest must reflect the STREAM. An op nothing emitted must not
    /// appear just because the emitter could have emitted it.
    #[test]
    fn opcodes_come_from_the_stream() {
        let man = build(&model(), "sm_120a");
        let ops: Vec<&str> = man["opcodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(ops.contains(&"FlashDecodeFp8"));
        assert!(ops.contains(&"Gemv"));
        assert!(!ops.contains(&"FlashMlaDecode"));
    }

    #[test]
    fn devblob_input_scope_is_explicitly_text_only() {
        let man = build(&model(), "gfx950");
        assert_eq!(man["input_contract"]["kind"], "token_ids");
        assert_eq!(man["input_contract"]["modalities"], json!(["text"]));
        assert_eq!(man["input_contract"]["vision"], false);
    }

    #[test]
    fn expert_parallel_objects_and_geometry_come_from_packet_immediates() {
        let mut m = model();
        m.progs[0].insts.extend([
            inst(DevOp::MoeGroupGluPf, [3072, 3584, 896, 2, 0, 1, 8, 0]),
            inst(DevOp::MoeGroupDownPf, [3584, 3072, 896, 2, 0, 0, 8, 0]),
        ]);
        let man = build(&m, "gfx950");
        let ep = &man["objects"]["lean"]["moe_prefill_ep"];
        assert_eq!(ep["required"], true);
        assert_eq!(ep["boundaries"][0]["degree"], 8);
        assert_eq!(ep["boundaries"][0]["experts"], 896);
        assert_eq!(ep["boundaries"][0]["full_intermediate_width"], 3072);
        assert_eq!(ep["resource_contract"]["wavefront_size"], 64);
        assert_eq!(ep["resource_contract"]["private_segment_bytes"], 0);
        assert_eq!(
            ep["additional_resident_bytes_per_rank"],
            112u64 * 3 * (3584u64 * 3072 / 2 + 3584u64 * (3072 / 32))
        );
        assert_eq!(
            ep["capacity_ack_env"],
            "PLOW_MOE_PREFILL_EP_MAX_EXTRA_BYTES"
        );
        assert!(config_header(&man).contains("#define PLOW_PACKET_REQUIRES_MOE_PREFILL_EP 1"));

        let ordinary = build(&model(), "gfx950");
        assert_eq!(
            ordinary["objects"]["lean"]["moe_prefill_ep"]["required"],
            false
        );
        assert!(config_header(&ordinary).contains("#define PLOW_PACKET_REQUIRES_MOE_PREFILL_EP 0"));
    }

    /// The opt-in lean MoE body variants are object requests carried by the config header;
    /// the default manifest carries neither key nor macro, so its pairing hash is unchanged.
    #[test]
    fn moe_body_variants_are_opt_in_object_requests() {
        let _guard = crate::test_env::env_guard();
        let ordinary = build(&model(), "gfx950");
        assert!(ordinary["objects"]["lean"].get("moe_stage1_body").is_none());
        assert!(ordinary["objects"]["lean"].get("moe_stage2_body").is_none());
        assert!(!config_header(&ordinary).contains("PLOW_OBJECT_MOE_STAGE"));
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_MOE_STAGE1_BODY", "1"),
            ("PLOW_MOE_STAGE2_BODY", "1"),
        ]);
        let man = build(&model(), "gfx950");
        assert_eq!(man["objects"]["lean"]["moe_stage1_body"]["required"], true);
        assert_eq!(man["objects"]["lean"]["moe_stage2_body"]["required"], true);
        let header = config_header(&man);
        assert!(header.contains("#define PLOW_OBJECT_MOE_STAGE1_BODY 1\n"));
        assert!(header.contains("#define PLOW_OBJECT_MOE_STAGE2_BODY 1\n"));
        assert_ne!(man["pairing"]["hash"], ordinary["pairing"]["hash"]);
    }

    /// L4: a decode XReduce with `i7 != 0` (the folded latent combine) is a build axis of the
    /// tagged decode object; the ordinary model must not carry it.
    #[test]
    fn xr_combine_fold_is_a_decode_object_requirement() {
        let mut m = model();
        m.progs[1]
            .insts
            .push(inst(DevOp::XReduce, [3584, 8, 0, 5, 0, 0, 0, 16]));
        let man = build(&m, "gfx950");
        assert_eq!(man["features"]["xr_combine_fold"], true);
        let req: Vec<&str> = man["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(req.contains(&"PLOW_XR_COMBINE_FOLD=1"), "{req:?}");
        let hdr = config_header(&man);
        assert!(hdr.contains("#define PLOW_PACKET_REQUIRES_XR_COMBINE_FOLD 1\n#ifndef PLOW_XR_COMBINE_FOLD\n#define PLOW_XR_COMBINE_FOLD 1\n#endif\n"));

        let ordinary = build(&model(), "gfx950");
        assert_eq!(ordinary["features"]["xr_combine_fold"], false);
        let req: Vec<&str> = ordinary["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(!req.contains(&"PLOW_XR_COMBINE_FOLD=1"));
        assert!(config_header(&ordinary).contains("#define PLOW_PACKET_REQUIRES_XR_COMBINE_FOLD 0\n#ifndef PLOW_XR_COMBINE_FOLD\n#define PLOW_XR_COMBINE_FOLD 0\n#endif\n"));
    }

    /// L3: a `KdaStateStepG` with flags bit 2 (the folded f_b GEMV) is a build axis of the
    /// decode object; the ordinary model must not carry it.
    #[test]
    fn kda_fb_fold_is_a_decode_object_requirement() {
        let mut m = model();
        m.progs[1]
            .insts
            .push(inst(DevOp::KdaStateStepG, [1, 12, 128, 16, 5, 0, 1, 0]));
        let man = build(&m, "gfx950");
        assert_eq!(man["features"]["kda_fb_fold"], true);
        let req: Vec<&str> = man["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(req.contains(&"PLOW_KDA_FB_FOLD=1"), "{req:?}");
        assert!(config_header(&man).contains("#define PLOW_PACKET_REQUIRES_KDA_FB_FOLD 1\n#ifndef PLOW_KDA_FB_FOLD\n#define PLOW_KDA_FB_FOLD 1\n#endif\n"));

        let mut plain = model();
        plain.progs[1]
            .insts
            .push(inst(DevOp::KdaStateStepG, [1, 12, 128, 16, 1, 0, 1, 0]));
        let man = build(&plain, "gfx950");
        assert_eq!(man["features"]["kda_fb_fold"], false);
        let req: Vec<&str> = man["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(!req.contains(&"PLOW_KDA_FB_FOLD=1"));
        assert!(config_header(&man).contains("#define PLOW_PACKET_REQUIRES_KDA_FB_FOLD 0\n#ifndef PLOW_KDA_FB_FOLD\n#define PLOW_KDA_FB_FOLD 0\n#endif\n"));
    }

    #[test]
    fn kda_decode_fused_arm_is_a_decode_object_requirement() {
        let mut m = model();
        m.progs[1]
            .insts
            .push(inst(DevOp::KdaStateStepG, [1, 12, 128, 8, 9, 0, 1, 0]));
        let man = build(&m, "gfx950");
        assert_eq!(man["features"]["kda_decode_fused_arm"], true);
        assert_eq!(man["features"]["kda_fb_fold"], false);
        let req: Vec<&str> = man["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(req.contains(&"PLOW_KDA_DECODE_FUSED_ARM=1"), "{req:?}");
        assert!(!req.contains(&"PLOW_KDA_FB_FOLD=1"));
        let h = config_header(&man);
        assert!(h.contains("#define PLOW_PACKET_REQUIRES_KDA_DECODE_FUSED_ARM 1\n#ifndef PLOW_KDA_DECODE_FUSED_ARM\n#define PLOW_KDA_DECODE_FUSED_ARM 1\n#endif\n"));
        // L8 is packet-inert and defaults off in the header unless the emit asked for it.
        assert_eq!(man["features"]["gemv_prefetch"], false);
        assert!(h.contains("#ifndef PLOW_GEMV_PREFETCH\n#define PLOW_GEMV_PREFETCH 0\n#endif\n"));

        let mut both = model();
        both.progs[1]
            .insts
            .push(inst(DevOp::KdaStateStepG, [1, 12, 128, 8, 13, 0, 1, 0]));
        let man = build(&both, "gfx950");
        assert_eq!(man["features"]["kda_decode_fused_arm"], true);
        assert_eq!(man["features"]["kda_fb_fold"], true);

        let plain = model();
        let man = build(&plain, "gfx950");
        assert_eq!(man["features"]["kda_decode_fused_arm"], false);
        assert!(config_header(&man).contains("#define PLOW_PACKET_REQUIRES_KDA_DECODE_FUSED_ARM 0\n#ifndef PLOW_KDA_DECODE_FUSED_ARM\n#define PLOW_KDA_DECODE_FUSED_ARM 0\n#endif\n"));
    }

    /// One opcode, two bodies: hd is an instruction field, so hd256 and hd512
    /// must be SEPARATE arms or a specialised object drops one of them.
    #[test]
    fn flash_arms_split_by_head_dim() {
        let man = build(&model(), "sm_120a");
        let union: Vec<&str> = man["union"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(union.contains(&"FlashDecode/hd256"));
        assert!(union.contains(&"FlashDecodeFp8/hd512"));
    }

    /// `GV_MM_MAX = next_pow2(decode_batch)` — the −19.4% bug. B=8 must not
    /// produce 16.
    #[test]
    fn gv_mm_max_follows_decode_batch() {
        let man = build(&model(), "sm_120a");
        assert_eq!(man["shapes"]["decode_batch"], 8);
        assert_eq!(man["tuning"]["gv_mm_max"], 8);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(16), 16);
    }

    /// `GF_FULL = gqa` only when the full layers have one KV head — the 1.48x.
    #[test]
    fn gf_full_follows_gqa_when_one_kv_head() {
        let man = build(&model(), "sm_120a");
        assert_eq!(man["shapes"]["gqa"], 8);
        assert_eq!(man["tuning"]["gf_full"], 8);
    }

    /// Full layers with >1 KV head: the rule does not apply and must stay silent
    /// rather than guess.
    #[test]
    fn gf_full_absent_when_kv_heads_gt_one() {
        let dec = prog(vec![inst(DevOp::FlashDecode, [8, 8, 2, 0, 0, 0, 512, 0])]);
        let m = Model {
            n_cu: 170,
            target: 0,
            tensors: vec![],
            progs: vec![dec],
            kv_row_insts: vec![],
            prog_t: vec![8],
            gen: vec![],
        };
        assert!(build(&m, "sm_120a")["tuning"].get("gf_full").is_none());
    }

    /// Per-program arm sets, keyed on (kind, bucket|batch, segment).
    #[test]
    fn programs_are_listed_per_bucket() {
        let man = build(&model(), "sm_120a");
        let p = man["programs"].as_array().unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0]["kind"], "prefill");
        assert_eq!(p[0]["bucket"], 1024);
        assert_eq!(p[1]["kind"], "decode");
        assert_eq!(p[1]["batch"], 8);
        assert!(p[0]["segment"].is_null());
    }

    /// The nvcc rendering is a BACKEND of the neutral facts, and `requires` is
    /// the correctness half.
    #[test]
    fn nvcc_backend_renders_required_flags() {
        let man = build(&model(), "sm_120a");
        let req: Vec<&str> = man["backends"]["nvcc"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(req.contains(&"PLOW_FP8_KV=1"));
        assert!(req.contains(&"PLOW_NV_PREFILL=1"));
        assert!(!req.contains(&"PLOW_NV_PF_GEMV_HEAD=1"));
        assert!(!req.contains(&"PLOW_NV_W8A8=1"));
        let rec: Vec<&str> = man["backends"]["nvcc"]["recommends"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(rec.contains(&"GV_MM_MAX=8"));
        assert!(rec.contains(&"PLOW_NV_FA_GF_FULL=8"));
    }

    #[test]
    fn nvcc_prefill_gemv_requirement_follows_program_phase() {
        let mut m = model();
        m.progs[0].insts.push(inst(DevOp::Gemv, [0; 8]));
        let man = build(&m, "sm_120a");
        let req = man["backends"]["nvcc"]["requires"].as_array().unwrap();
        assert!(req.iter().any(|v| v == "PLOW_NV_PF_GEMV_HEAD=1"));
    }

    /// The pairing hash must move when the compiled arm set moves, and must NOT
    /// move for a cosmetic manifest change.
    #[test]
    fn pairing_hash_tracks_the_compiled_set() {
        let a = build(&model(), "sm_120a");
        let mut b = a.clone();
        assert_eq!(pairing_hash(&a), pairing_hash(&b));
        b["analysis"] = json!("something else entirely");
        assert_eq!(pairing_hash(&a), pairing_hash(&b));
        b["union"] = json!(["Gemv"]);
        assert_ne!(pairing_hash(&a), pairing_hash(&b));
    }

    /// The four axes must be readable WITHOUT reconstructing them from the feature booleans —
    /// that reconstruction is a judgement, and it was being made separately in every consumer.
    #[test]
    fn precision_axes_are_a_lookup_not_a_judgement() {
        let man = build(&model(), "gfx950");
        let p = &man["precision"];
        // The fixture is bf16 weights, bf16 activations, fp8 KV (FlashDecodeFp8), no experts.
        assert_eq!(p["weight_enc"], "bf16");
        assert_eq!(p["act_enc"], "bf16", "no QuantFp8 => wide activations");
        assert_eq!(
            p["kv_enc"], "fp8",
            "the KV axis is INDEPENDENT of the weight axis"
        );
        assert_eq!(p["expert_enc"], "none");
    }

    /// The activation axis is `QuantFp8`'s presence — the op that IS the activation quant — and
    /// not an inference from the weight flag. That distinction is the whole point: the axis had no
    /// flag and was decided by phase, which is how a w8a16 packet reached a w8a8-only object.
    #[test]
    fn activation_axis_follows_quant_fp8_not_the_weight_flag() {
        let w8a16 = prog(vec![inst(DevOp::GemmFp8, [0; 8])]);
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![w8a16],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        };
        let man = build(&m, "gfx950");
        assert_eq!(man["precision"]["weight_enc"], "fp8");
        assert_eq!(
            man["precision"]["act_enc"], "bf16",
            "fp8 weights, bf16 activations = w8a16"
        );

        let w8a8 = prog(vec![
            inst(DevOp::QuantFp8, [0; 8]),
            inst(DevOp::GemmFp8, [0; 8]),
        ]);
        let m2 = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![w8a8],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        };
        let man2 = build(&m2, "gfx950");
        assert_eq!(man2["precision"]["act_enc"], "fp8");
        assert_eq!(man2["features"]["w8a8"], true);
    }

    /// The gfx950 backend renders the defines a covering object must carry. On AMD a missing arm
    /// does not trap — it writes nothing — so `requires` is the correctness half in a stronger
    /// sense than on NVIDIA.
    #[test]
    fn gfx950_backend_renders_the_axis_defines() {
        let man = build(&model(), "gfx950");
        let req: Vec<&str> = man["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            req.contains(&"PLOW_BUCKET_DECODE=0"),
            "the packet has a prefill bucket"
        );
        assert!(req.contains(&"PLOW_FP8_KV=1"));
        assert!(
            !req.contains(&"PLOW_FP8=1"),
            "bf16 weights must not ask for the fp8 object"
        );
        assert!(
            !req.contains(&"PLOW_MLA_PREFILL=1"),
            "no MLA ops in this packet"
        );
        assert!(
            !req.contains(&"PLOW_K3=1"),
            "no K3 block ops in this packet either"
        );
        // nvcc is still rendered — adding a backend must not remove one.
        assert!(man["backends"]["nvcc"]["requires"].is_array());
    }

    /// A `QuantFp8` carrying `t[3]` must ask for `PLOW_T11_GLUQUANT=1`, and a plain one must not.
    ///
    /// THE REGRESSION THIS PINS. `qnorm_fuse` folds the GLU producer into the quant packet
    /// (t3=gate, t4=up, i2=act) and DELETES the `Glu` packet that used to compute `fu`. The AMD
    /// dispatch ignored t3/t4 for its whole life, so the packet quantized an `fu` nothing had
    /// written: no fault, no NaN, just a garbage FFN output, a wrong KV cache, and fluent wrong
    /// tokens. `gfx950_coverage_tests::dispatched_list_matches_the_amd_interpreter` could not see
    /// it — the OPCODE was dispatched, only its operands were dropped — which is why the pairing
    /// is stated here, at the field, and refused by `PREFILL_ARM_MARKERS` at load.
    #[test]
    fn a_folded_quant_packet_requires_the_glu_arm() {
        let req = |t3: u32| -> Vec<String> {
            let mut q = inst(DevOp::QuantFp8, [0; 8]);
            q.t = [0, 1, 2, t3, 4, 0, 0, 0];
            let m = Model {
                n_cu: 256,
                target: 0,
                tensors: vec![],
                progs: vec![prog(vec![q, inst(DevOp::GemmFp8, [0; 8])])],
                kv_row_insts: vec![],
                prog_t: vec![128],
                gen: vec![],
            };
            build(&m, "gfx942")["backends"]["gfx942"]["requires"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        };
        assert!(req(3).contains(&"PLOW_T11_GLUQUANT=1".to_string()));
        assert!(!req(packet::TENSOR_NONE).contains(&"PLOW_T11_GLUQUANT=1".to_string()));
    }

    /// A packet carrying Kimi-K3's BLOCK ops must ask for `PLOW_K3=1`.
    ///
    /// It is a MODEL axis, not a prefill or precision one, and it is the arm flag whose absence is
    /// most completely silent. `AttnRes` (104) replaces the residual ADD twice in every layer,
    /// `SituGlu` (105) is the activation on every GLU, `MlaOutGate` (106) gates 24 of 93 layers and
    /// the KDA recurrence (99-103) is the other 69 mixers. An object built without the flag has no
    /// `case` for any of them, and this interpreter's dispatch `default:` does not trap — it writes
    /// NOTHING. The packet runs, the buffers keep what they held, and a model missing most of
    /// itself returns fluent output.
    ///
    /// Any ordinary K3 opcode is enough to require the family include; the generated per-object
    /// presence macros then prune individual bodies. A `K3_NLAYERS=3` truncation emits no MLA layer and
    /// therefore no `MlaOutGate`; a decode-only blob emits no prefill op at all. Both still need
    /// the flag.
    #[test]
    fn a_k3_packet_requires_the_k3_arms() {
        let gfx = |ops: &[DevOp]| -> Vec<String> {
            let p = prog(ops.iter().map(|&o| inst(o, [0; 8])).collect());
            let m = Model {
                n_cu: 256,
                target: 0,
                tensors: vec![],
                progs: vec![p],
                kv_row_insts: vec![],
                prog_t: vec![1],
                gen: vec![],
            };
            build(&m, "gfx950")["backends"]["gfx950"]["requires"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        };
        for op in [
            DevOp::AttnRes,
            DevOp::SituGlu,
            DevOp::MlaOutGate,
            DevOp::KdaStateStep,
            DevOp::KdaStateStepG,
            DevOp::KdaConv,
            DevOp::KdaConv3,
            DevOp::KdaGatedNorm,
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ] {
            assert!(
                gfx(&[op]).iter().any(|r| r == "PLOW_K3=1"),
                "{op:?} alone must still require the K3 object"
            );
        }
        // And it does not leak onto a packet that has none of them.
        assert!(!gfx(&[DevOp::Gemv, DevOp::RmsNorm])
            .iter()
            .any(|r| r == "PLOW_K3=1"));

        let chunk_req = gfx(&[DevOp::KdaChunkCarry]);
        assert!(chunk_req.iter().any(|r| r == "PLOW_KDA_CHUNK=1"));
        assert!(!gfx(&[DevOp::KdaStateStepG])
            .iter()
            .any(|r| r == "PLOW_KDA_CHUNK=1"));

        let qpre_req = |enabled: bool| -> Vec<String> {
            let mut carry = inst(DevOp::KdaChunkCarry, [0; 8]);
            carry.i[4] = u32::from(enabled);
            let m = Model {
                n_cu: 256,
                target: 0,
                tensors: vec![],
                progs: vec![prog(vec![carry])],
                kv_row_insts: vec![],
                prog_t: vec![8192],
                gen: vec![],
            };
            build(&m, "gfx950")["backends"]["gfx950"]["requires"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        };
        assert!(qpre_req(true).iter().any(|r| r == "PLOW_KDA_CHUNK_QPRE=1"));
        assert!(!qpre_req(false).iter().any(|r| r == "PLOW_KDA_CHUNK_QPRE=1"));
    }

    #[test]
    fn every_ordinary_k3_dispatch_is_presence_gated() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("runtime/amd/interp.hip");
        let Ok(src) = std::fs::read_to_string(path) else {
            return;
        };
        for op in [
            DevOp::KdaConv,
            DevOp::KdaGate,
            DevOp::KdaStateStep,
            DevOp::KdaGatedNorm,
            DevOp::AttnRes,
            DevOp::SituGlu,
            DevOp::MlaOutGate,
            DevOp::KdaConv3,
            DevOp::KdaStateStepG,
            DevOp::KdaConvStateStepG,
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ] {
            let case = format!("        case {}:", op.c_name());
            let guard = format!(
                "#if {}\n{case}",
                op.c_name().replace("PLOW_DOP_", "PLOW_HAS_")
            );
            assert_eq!(
                src.matches(&case).count(),
                src.matches(&guard).count(),
                "every {} dispatch occurrence must be directly presence-gated",
                op.c_name(),
            );
        }
    }

    #[test]
    fn every_prunable_decode_dispatch_is_presence_gated() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("runtime/amd/interp.hip");
        let Ok(src) = std::fs::read_to_string(path) else {
            return;
        };
        for op in [
            DevOp::RmsNorm,
            DevOp::RowRms,
            DevOp::LayerNorm,
            DevOp::HeadNormRope,
            DevOp::NormResidual,
            DevOp::Residual,
            DevOp::AddNorm,
            DevOp::NormResidualNorm,
            DevOp::Glu,
            DevOp::Embed,
            DevOp::SoftCap,
            DevOp::Argmax,
            DevOp::ArgmaxFin,
            DevOp::Gemv,
            DevOp::GemvGlu,
            DevOp::GemvQkv,
            DevOp::GemvQkvg,
            DevOp::FlashDecode,
            DevOp::FlashMlaDecode,
            DevOp::OUvFold,
            DevOp::MlaMergeFold,
            DevOp::AttnSelect,
            DevOp::IndexScore,
            DevOp::IndexSelect,
            DevOp::MoeRouter,
            DevOp::MoeExpertGlu,
            DevOp::MoeExpertDown,
            DevOp::MoeExpertGluFp8Blk,
            DevOp::MoeExpertDownFp8Blk,
            DevOp::DenseGluFp8Blk,
            DevOp::GemvFp8Blk,
            DevOp::FlashMerge,
        ] {
            let case = format!("        case {}:", op.c_name());
            let guard = format!(
                "#if !PLOW_DECODE_INVENTORY_PRUNE || {}\n{case}",
                op.c_name().replace("PLOW_DOP_", "PLOW_HAS_")
            );
            assert!(
                src.contains(&guard),
                "{} is not directly guarded by its decode inventory bit",
                op.c_name()
            );
        }
    }

    #[test]
    fn every_prunable_prefill_dispatch_is_presence_gated() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("runtime/amd/interp.hip");
        let src = std::fs::read_to_string(path).unwrap();
        for op in [
            DevOp::Gemv,
            DevOp::Gemm,
            DevOp::GemmSmall,
            DevOp::GemmMed,
            DevOp::GemmWide,
            DevOp::GemmC5,
            DevOp::GemmFp8Blk,
            DevOp::GemmMxfp4,
            DevOp::GemmMedMxfp4,
            DevOp::GemmSmallMxfp4,
            DevOp::GemmWideMxfp4,
            DevOp::GemmC5Mxfp4,
            DevOp::IndexScorePf,
            DevOp::IndexSelectPf,
            DevOp::IndexUnionPf,
            DevOp::OUvFold,
            DevOp::FlashMerge,
        ] {
            let case = format!("        case {}:", op.c_name());
            let guard = format!(
                "#if !PLOW_DECODE_INVENTORY_PRUNE || {}\n{case}",
                op.c_name().replace("PLOW_DOP_", "PLOW_HAS_")
            );
            assert!(
                src.contains(&guard),
                "{} is not inventory-gated",
                op.c_name()
            );
        }
    }

    #[test]
    fn k3_object_inventory_covers_every_ordinary_and_lean_arm() {
        let ordinary = [
            DevOp::KdaConv,
            DevOp::KdaGate,
            DevOp::KdaStateStep,
            DevOp::KdaGatedNorm,
            DevOp::AttnRes,
            DevOp::SituGlu,
            DevOp::MlaOutGate,
            DevOp::KdaConv3,
            DevOp::KdaStateStepG,
            DevOp::KdaConvStateStepG,
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkIntra,
            DevOp::KdaChunkWu,
            DevOp::KdaChunkCarry,
        ];
        let make = || prog(ordinary.iter().map(|&op| inst(op, [0; 8])).collect());
        let mut decode = make();
        decode.insts.push(inst(DevOp::KdaDecodeFused, [0; 8]));
        let m = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![make(), decode],
            kv_row_insts: vec![],
            prog_t: vec![128, 1],
            gen: vec![],
        };
        let man = build(&m, "gfx950");
        for phase in ["prefill", "decode"] {
            let arms = man["objects"]["ordinary"][phase]["arms"]
                .as_array()
                .unwrap();
            for op in ordinary {
                let name = op_name(op);
                assert!(
                    arms.iter()
                        .filter_map(Value::as_str)
                        .any(|a| a.split('/').next() == Some(&name)),
                    "{phase} inventory misses {name}"
                );
            }
        }
        let flash = man["objects"]["ordinary"]["flash"]["arms"]
            .as_array()
            .unwrap();
        assert!(
            flash.is_empty(),
            "a K3-only graph must not pull heavy K3 bodies into flash"
        );
        assert_eq!(man["objects"]["lean"]["kda_decode_fused"]["required"], true);
        assert!(man["objects"]["lean"]["kda_decode_fused"]["arms"][0]
            .as_str()
            .unwrap()
            .starts_with("KdaDecodeFused/abi"));
        let h = config_header(&man);
        assert!(h.contains("#define PLOW_PACKET_HAS_KDA_DECODE_FUSED 1"));
        assert!(h.contains("#if defined(PLOW_BUCKET_FLASH)\n#define PLOW_HAS_ATTN_RES 0"));

        let decode = &man["objects"]["ordinary"]["decode"];
        assert_eq!(decode["inventory_prune_capable"], true);
        assert_eq!(decode["resource_contract"]["max_total_registers"], 256);
        assert_eq!(
            decode["resource_contract"]["min_occupancy_waves_per_simd"],
            2
        );
        assert!(decode["families"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "kda"));
        assert!(decode["excluded_opcodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "FlashPrefill"));

        let chains = man["dispatch_chains"].as_array().unwrap();
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0]["program"], 0);
        assert_eq!(chains[0]["kind"], "prefill");
        assert_eq!(chains[0]["bucket"], 128);
        assert_eq!(chains[0]["aql_replay"]["packets"], 1);
        assert_eq!(chains[0]["phases"][0]["first_segment"], 0);
        assert_eq!(chains[0]["phases"][0]["last_segment"], 0);
        assert_eq!(
            chains[0]["aql_replay"]["rank_commit"],
            "prepare_all_ranks_then_ring_all"
        );
        assert_eq!(
            chains[0]["segments"][0]["resource_contract"]["max_private_segment_bytes_delta"],
            0
        );
        assert_eq!(chains[1]["batch"], 1);
    }

    #[test]
    fn opcode_family_partition_is_model_neutral_and_stable() {
        for (op, family) in [
            ("KdaChunkCarry", "kda"),
            ("MoeGroupDownPf", "moe"),
            ("FlashMlaDecode", "mla"),
            ("FlashDecode", "attention"),
            ("GemvQkvg", "linear"),
            ("NormResidual", "norm_residual"),
            ("XReduceTwoShot", "collective"),
            ("XReduceScatter", "collective"),
            ("XAllGather", "collective"),
            ("Argmax", "elementwise"),
        ] {
            assert_eq!(opcode_family(op), family, "{op}");
        }
        let p = prog(vec![inst(DevOp::Gemv, [0; 8])]);
        let m = Model {
            n_cu: 1,
            target: 0,
            tensors: vec![],
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        assert!(
            build(&m, "sm_120a")["objects"]["ordinary"]["decode"]["resource_contract"].is_null()
        );
    }

    #[test]
    fn kda_key_factor_pair_changes_the_derived_object_inventory() {
        let arm = |op: &str| Arm {
            op: op.into(),
            hd: None,
            variant: Some("d128_qpre".into()),
        };
        let segment = |op: &str| ProgramArms {
            program: 0,
            kind: "prefill",
            packed_prefill_only: false,
            t: 8192,
            seg: Some(1),
            arms: BTreeSet::from([arm(op)]),
            insts: 1,
        };
        let wu = segment("KdaChunkWu");
        let carry = segment("KdaChunkCarry");

        let incomplete = object_inventory(std::slice::from_ref(&wu), "gfx950");
        assert_eq!(incomplete["lean"]["kda_key_factor_pair"]["required"], false);
        let paired = object_inventory(&[wu, carry], "gfx950");
        assert_eq!(paired["lean"]["kda_key_factor_pair"]["required"], true);
        assert_eq!(
            paired["lean"]["kda_key_factor_pair"]["wu_arms"][0],
            "KdaChunkWu/d128_qpre"
        );
        assert_eq!(
            paired["lean"]["kda_key_factor_pair"]["carry_arms"][0],
            "KdaChunkCarry/d128_qpre"
        );
        assert_ne!(incomplete, paired);
    }

    #[test]
    fn decode_mla_object_requires_one_pure_two_instruction_pair() {
        let arm = |op: &str| Arm {
            op: op.into(),
            hd: None,
            variant: None,
        };
        let segment = |insts, ops: &[&str]| ProgramArms {
            program: 0,
            kind: "decode",
            packed_prefill_only: false,
            t: 1,
            seg: Some(1),
            arms: ops.iter().map(|op| arm(op)).collect(),
            insts,
        };
        let pure = segment(2, &["FlashMlaDecode", "MlaMergeFold"]);
        let inv = object_inventory(std::slice::from_ref(&pure), "gfx950");
        assert_eq!(inv["ordinary"]["decode_mla"]["required"], true);
        let manifest = json!({"union": [], "objects": inv});
        assert!(config_header(&manifest).contains("#define PLOW_PACKET_HAS_DECODE_MLA_SEGMENTS 1"));

        for rejected in [
            segment(1, &["FlashMlaDecode"]),
            segment(3, &["FlashMlaDecode", "MlaMergeFold", "Gemv"]),
        ] {
            assert_eq!(
                object_inventory(&[rejected], "gfx950")["ordinary"]["decode_mla"]["required"],
                false
            );
        }
    }

    #[test]
    fn direct_gfx950_build_tracks_decode_mla_inventory() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("scripts/build_gfx950.sh");
        let script = std::fs::read_to_string(path).unwrap();
        for required in [
            "#define PLOW_PACKET_HAS_DECODE_MLA_SEGMENTS 1",
            "interp_decode_mla.elf",
            "interp_decode_mla_gq.elf",
            "check decode_mla ",
            "check decode_mla_gq ",
            "plow_decode_mla_segment_object_1",
            "plow_packet_hash_lo",
            "plow_packet_hash_hi",
            "$DECODE_MLA_ELFS",
        ] {
            assert!(
                script.contains(required),
                "direct gfx950 build misses {required}"
            );
        }
    }

    /// `arm_of` must read each flash op's head-dim from the slot that op ACTUALLY carries it in.
    ///
    /// `FlashMerge` uses `i[3]` (`runtime/amd/interp.hip:1119` / `:1315`), the rest use `i[6]`.
    /// Reading `i[6]` for `FlashMerge` yields `Some(0)` on every real packet, so the whole
    /// coverage check goes blind for `d_flash_merge<D>` — the exact template family a dispatch
    /// bug has already been found in once. This is a constant-vs-constant comparison that always
    /// passes, which is why it needs its own test rather than being caught downstream.
    #[test]
    fn arm_of_reads_the_head_dim_slot_each_flash_op_actually_uses() {
        let mut i = [0u32; 8];
        i[2] = 128; // HeadNormRope's slot
        i[3] = 256; // FlashMerge's slot
        i[6] = 128; // FlashPrefill / FlashDecode's slot
        assert_eq!(
            arm_of(DevOp::FlashMerge, &i).hd,
            Some(256),
            "FlashMerge takes i[3]"
        );
        assert_eq!(arm_of(DevOp::FlashPrefill, &i).hd, Some(128));
        assert_eq!(arm_of(DevOp::FlashDecode, &i).hd, Some(128));
        assert_eq!(arm_of(DevOp::FlashDecodeFp8, &i).hd, Some(128));
        assert_eq!(arm_of(DevOp::HeadNormRope, &i).hd, Some(128));
        assert_eq!(arm_of(DevOp::HeadNormRopeFp8, &i).hd, Some(128));
        assert_eq!(
            arm_of(DevOp::Gemv, &i).hd,
            None,
            "non-flash ops are not templated on a field"
        );
        // A real FlashMerge packet leaves i[6] at 0, so the old slot cannot distinguish arms.
        let mut real = [0u32; 8];
        real[3] = 512;
        assert_eq!(arm_of(DevOp::FlashMerge, &real).hd, Some(512));
    }

    #[test]
    fn gemm_wide_manifest_preserves_the_selected_tile() {
        let plain = arm_of(DevOp::GemmWide, &[0; 8]);
        assert_eq!(plain.key(), "GemmWide");

        let mut tagged = [0; 8];
        tagged[7] = GEMM_WIDE_C8_TAG;
        assert_eq!(
            arm_of(DevOp::GemmWide, &tagged).key(),
            "GemmWide/tile128x384x64"
        );
    }

    #[test]
    fn decode_header_is_an_exact_opcode_and_headnorm_inventory() {
        let mut hn = inst(DevOp::HeadNormRope, [0; 8]);
        hn.i[2] = 64;
        let decode_ops = [
            DevOp::RmsNorm,
            DevOp::HeadNormRope,
            DevOp::Embed,
            DevOp::Gemv,
            DevOp::Argmax,
            DevOp::ArgmaxFin,
            DevOp::GemvGlu,
            DevOp::XReduce,
            DevOp::MoeCombine,
            DevOp::MoeGroupGluFp8Blk,
            DevOp::MoeGroupDownFp8Blk,
            DevOp::FlashMlaDecode,
            DevOp::MoeRouterTopk,
            DevOp::MlaMergeFold,
            DevOp::KdaGatedNorm,
            DevOp::AttnRes,
            DevOp::SituGlu,
            DevOp::MlaOutGate,
            DevOp::GemvQkvg,
            DevOp::KdaConv3,
            DevOp::KdaStateStepG,
        ];
        let mut insts = vec![hn];
        insts.extend(
            decode_ops
                .iter()
                .copied()
                .filter(|op| *op != DevOp::HeadNormRope)
                .map(|op| inst(op, [0; 8])),
        );
        let model = Model {
            n_cu: 256,
            target: 0,
            tensors: vec![],
            progs: vec![prog(insts)],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        let header = config_header(&build(&model, "gfx950"));
        for op in DevOp::ALL {
            let macro_name = op.c_name().replace("PLOW_DOP_", "PLOW_HAS_");
            let expected = if decode_ops.contains(op) { 1 } else { 0 };
            let needle = format!("#elif PLOW_BUCKET_DECODE\n#define {macro_name} {expected}\n");
            assert!(
                header.contains(&needle),
                "decode inventory has the wrong presence value for {op:?}"
            );
        }
        assert!(header.contains("#define PLOW_HAS_HEADNORM_HD64 1"));
        for hd in [128, 256, 512] {
            assert!(header.contains(&format!("#define PLOW_HAS_HEADNORM_HD{hd} 0")));
        }
    }

    /// The header gates arms on presence, and the guard lets an explicit -D win.
    #[test]
    fn header_has_presence_and_shape_macros() {
        let h = config_header(&build(&model(), "sm_120a"));
        assert!(h.contains("#define PLOW_HAS_FLASH_DECODE_FP8 1"));
        assert!(h.contains("#define PLOW_HAS_FLASH_MLA_DECODE 0"));
        assert!(h.contains("#define PLOW_HAS_FLASH_HD512 1"));
        assert!(h.contains("#define GV_MM_MAX 8"));
        assert!(h.contains("#ifndef GV_MM_MAX"));
        assert!(h.contains("PLOW_PACKET_HASH"));
    }

    #[test]
    fn header_derives_chunk_qpre_from_the_exact_arm_variant() {
        let mut qpre = build(&model(), "gfx950");
        qpre["union"] = json!(["KdaChunkWu/d128_qpre", "KdaChunkCarry/d128_qpre"]);
        assert!(config_header(&qpre).contains("#define PLOW_KDA_CHUNK_QPRE 1"));

        let mut ordinary = build(&model(), "gfx950");
        ordinary["union"] = json!(["KdaChunkWu/d128", "KdaChunkCarry/d128"]);
        assert!(config_header(&ordinary).contains("#define PLOW_KDA_CHUNK_QPRE 0"));
    }

    #[test]
    fn wave_item_marker_requires_the_paired_object() {
        let make = |marked: bool| {
            let mut p = prog(vec![inst(DevOp::KdaChunkIntra, [0; 8])]);
            p.stream.push(packet::dev::StreamEnt {
                flags: if marked {
                    packet::dev::SE_KDA_INTRA_WAVE_ITEMS
                } else {
                    0
                },
                ..Default::default()
            });
            Model {
                n_cu: 256,
                target: 0,
                tensors: vec![],
                progs: vec![p],
                kv_row_insts: vec![],
                prog_t: vec![8192],
                gen: vec![],
            }
        };

        let required = build(&make(true), "gfx950");
        assert_eq!(
            required["objects"]["lean"]["kda_intra_wave_items"]["required"],
            true
        );
        assert!(config_header(&required)
            .contains("#define PLOW_PACKET_REQUIRES_KDA_INTRA_WAVE_ITEMS 1"));

        let rollback = build(&make(false), "gfx950");
        assert_eq!(
            rollback["objects"]["lean"]["kda_intra_wave_items"]["required"],
            false
        );
        assert!(config_header(&rollback)
            .contains("#define PLOW_PACKET_REQUIRES_KDA_INTRA_WAVE_ITEMS 0"));
        assert_ne!(pairing_hash(&required), pairing_hash(&rollback));
    }

    #[test]
    fn materialized_residual_input_capability_comes_from_instruction_operands() {
        let make = |fused: bool| {
            let mut attn = inst(DevOp::AttnRes, [0; 8]);
            attn.t = [packet::TENSOR_NONE; 8];
            attn.i[5] = packet::dev::TENSOR_NONE_I;
            if fused {
                attn.t[6] = 1;
                attn.t[7] = 2;
            }
            Model {
                n_cu: 4,
                target: 0,
                tensors: vec![],
                progs: vec![prog(vec![attn])],
                kv_row_insts: vec![],
                prog_t: vec![1],
                gen: vec![],
            }
        };

        let plain = build(&make(false), "gfx950");
        assert_eq!(plain["features"]["materialized_residual_input"], false);
        assert!(config_header(&plain).contains("#define PLOW_MATERIALIZED_RESIDUAL_INPUT 0"));

        let fused = build(&make(true), "gfx950");
        assert_eq!(fused["features"]["materialized_residual_input"], true);
        assert!(config_header(&fused).contains("#define PLOW_MATERIALIZED_RESIDUAL_INPUT 1"));
        assert!(fused["backends"]["gfx950"]["requires"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "PLOW_MATERIALIZED_RESIDUAL_INPUT=1"));
    }

    // ===== The `lean` block. =====================================================
    //
    // The gate is default-on and DEGRADES, so "it did not run" is now a normal,
    // common outcome — which makes it indistinguishable from "it ran and passed"
    // unless the artifact says which. `tuning.tier` already cost this project a
    // long stretch of meaningless GLM prefill numbers by reporting `portable` for
    // both "analytical model chosen" and "nothing was ever measured".

    /// THE REGRESSION GUARD FOR CORRECTION 3. Binary absent ⇒ the field is
    /// present, false, and CARRIES A REASON. A `reason` of `null` here would
    /// mean a consumer sees `verified: false` with no way to tell a skip from
    /// anything else.
    #[test]
    fn lean_block_is_false_with_a_reason_when_the_verifier_is_absent() {
        let skipped = crate::LeanReport::skipped(
            "no `plow_verify` binary (set PLOW_VERIFY_BIN, or `lake build` in lean-plow/)",
        );
        let man = super::build(&model(), "gfx950", &skipped);
        let lean = man
            .get("lean")
            .expect("build.json must carry a `lean` block");
        assert_eq!(lean["verified"], json!(false));
        assert_eq!(lean["oracle"], json!(false));
        let reason = lean["reason"]
            .as_str()
            .expect("a skip must state its reason");
        assert!(
            reason.contains("plow_verify"),
            "reason should name the binary: {reason}"
        );
    }

    /// The other side of it: a real pass says so, and says it with no reason.
    #[test]
    fn lean_block_reports_a_clean_verification() {
        let ok = crate::LeanReport {
            verified: true,
            oracle: true,
            reason: None,
        };
        let man = super::build(&model(), "gfx950", &ok);
        assert_eq!(man["lean"]["verified"], json!(true));
        assert_eq!(man["lean"]["oracle"], json!(true));
        assert_eq!(man["lean"]["reason"], Value::Null);
    }

    /// The two subsystems are INDEPENDENT (Correction 1). Disabling the
    /// certificate must not report the oracle as skipped, or the manifest
    /// re-creates in data the CLI coupling the flags were fixed to avoid.
    #[test]
    fn oracle_and_verify_are_recorded_independently() {
        let oracle_only = crate::LeanReport {
            verified: false,
            oracle: true,
            reason: Some("ordering certificate disabled on the command line".into()),
        };
        let man = super::build(&model(), "gfx950", &oracle_only);
        assert_eq!(man["lean"]["verified"], json!(false));
        assert_eq!(man["lean"]["oracle"], json!(true));
    }

    /// VERIFICATION IS READ-ONLY AND MUST NOT MOVE THE PAIRING HASH.
    ///
    /// `pairing_hash` decides whether a packet and an interpreter object may be
    /// loaded together. If the lean block fed it, running on a box with a Lean
    /// build would produce a packet that refuses the object built on a box
    /// without one — a verification gate breaking a pairing it has no business
    /// touching. It is hashed over `union` + `tuning` only; this pins that.
    #[test]
    fn the_lean_block_does_not_change_the_pairing_hash() {
        let m = model();
        let a = super::build(&m, "gfx950", &crate::LeanReport::skipped("absent"));
        let b = super::build(
            &m,
            "gfx950",
            &crate::LeanReport {
                verified: true,
                oracle: true,
                reason: None,
            },
        );
        assert_ne!(a["lean"], b["lean"], "the test would be vacuous otherwise");
        assert_eq!(a["pairing"]["hash"], b["pairing"]["hash"]);
    }
}
