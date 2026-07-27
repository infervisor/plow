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

use packet::dev::DevOp;
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
}

impl Arm {
    pub fn key(&self) -> String {
        match self.hd {
            Some(hd) => format!("{}/hd{hd}", self.op),
            None => self.op.clone(),
        }
    }
}

/// Which shape field (if any) selects a template instantiation for this opcode.
///
/// ONLY the flash family is templated on an instruction field today: `d_flash_*`
/// is `<HD, GF>` and HD comes from `i[6]`. The GEMM tile variants are separate
/// OPCODES (`Gemm`/`GemmMed`/`GemmSmall`), so the opcode already carries them and
/// adding a second key would split one body into several phantom arms.
fn arm_of(op: DevOp, i: &[u32; 8]) -> Arm {
    let hd = matches!(
        op,
        DevOp::FlashPrefill
            | DevOp::FlashPrefillFp8
            | DevOp::FlashDecode
            | DevOp::FlashDecodeFp8
            | DevOp::FlashMerge
    )
    .then(|| i[6]);
    Arm { op: op_name(op), hd }
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
    pub kind: &'static str,
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
    // `Model::prog_t`'s last entry is the decode program; everything before it is
    // a prefill bucket. (`emit_dense_gqa` pushes the buckets then the decode.)
    let last = m.progs.len().saturating_sub(1);
    for (pi, p) in m.progs.iter().enumerate() {
        let kind = if pi == last { "decode" } else { "prefill" };
        let t = m.prog_t.get(pi).copied().unwrap_or(0);
        for (seg, arms) in segment_arms(p) {
            out.push(ProgramArms {
                kind,
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
    // `gq_seg_ofs` is `[n_seg+1]` bounds into `gq_stream`. `l2_domains != 0`
    // repurposes `seg` as an L2 DOMAIN (PLOW_NV_PLACE), not a wave-class segment
    // — partitioning by it there would be meaningless, so fall back to whole.
    let n_seg = p.gq_seg_ofs.len().saturating_sub(1);
    if p.l2_domains != 0 || n_seg <= 1 || p.gq_stream.is_empty() {
        let mut arms = BTreeSet::new();
        for inst in &p.insts {
            if let Some(op) = op_of(inst.op) {
                arms.insert(arm_of(op, &inst.i));
            }
        }
        return vec![(None, (arms, p.insts.len()))];
    }
    let mut out = Vec::new();
    for s in 0..n_seg {
        let (a, b) = (p.gq_seg_ofs[s] as usize, p.gq_seg_ofs[s + 1] as usize);
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
}

fn shapes(m: &Model) -> Shapes {
    let mut s = Shapes::default();
    let last = m.progs.len().saturating_sub(1);
    for (pi, p) in m.progs.iter().enumerate() {
        let decode = pi == last;
        if !decode {
            s.prefill_buckets.push(m.prog_t.get(pi).copied().unwrap_or(0));
        }
        for inst in &p.insts {
            let Some(op) = op_of(inst.op) else { continue };
            match op {
                // `i0=n_batch i1=n_head i2=n_kv_head … i6=hd`
                DevOp::FlashDecode | DevOp::FlashDecodeFp8 => {
                    let (hd, nh, kvh, nb) = (inst.i[6], inst.i[1], inst.i[2], inst.i[0]);
                    s.hd.insert(hd);
                    s.kv_heads.insert(kvh);
                    s.kv_dtype.insert(
                        hd,
                        if op == DevOp::FlashDecodeFp8 { "e4m3" } else { "bf16" },
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
                        .or_insert(if op == DevOp::FlashPrefillFp8 { "e4m3" } else { "bf16" });
                }
                _ => {}
            }
        }
    }
    s.max_chunk = s.prefill_buckets.iter().copied().max().unwrap_or(0);
    s
}

/// Neutral capability facts. Presence of an ARM implies the feature — the env
/// knob that produced it is not consulted.
fn features(union: &BTreeSet<Arm>) -> Map<String, Value> {
    let has = |n: &str| union.iter().any(|a| a.op == n);
    let mut f = Map::new();
    f.insert("fp8_kv".into(), json!(has("FlashDecodeFp8") || has("HeadNormRopeFp8")));
    f.insert(
        "fp8_weights".into(),
        json!(has("GemvFp8") || has("GemvGluFp8") || has("GemmFp8") || has("GemmGluFp8")),
    );
    // w8a8 is the per-row ACTIVATION quant: `QuantFp8` exists only on that path.
    f.insert("w8a8".into(), json!(has("QuantFp8")));
    f.insert("moe".into(), json!(union.iter().any(|a| a.op.starts_with("Moe"))));
    f.insert(
        "mla".into(),
        json!(has("FlashMlaDecode") || has("FlashMlaPrefill") || has("MlaMergeFold")),
    );
    f.insert("mamba".into(), json!(has("Mamba2Scan")));
    f.insert(
        "tensor_parallel".into(),
        json!(union.iter().any(|a| a.op.starts_with('X'))),
    );
    f.insert("prefill".into(), json!(has("FlashPrefill") || has("FlashPrefillFp8")));
    f
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
fn backend_nvcc(f: &Map<String, Value>, t: &Map<String, Value>) -> Value {
    let on = |k: &str| f.get(k).and_then(Value::as_bool).unwrap_or(false);
    let mut req = vec!["PLOW_NV_GEMMA=1".to_string()];
    if on("w8a8") {
        req.push("PLOW_NV_W8A8=1".into());
    }
    if on("fp8_kv") {
        req.push("PLOW_FP8_KV=1".into());
    }
    if on("prefill") {
        req.push("PLOW_NV_PREFILL=1".into());
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
            .filter(|q| q.kind == p.kind && (q.t != p.t || q.seg != p.seg))
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
pub fn build(m: &Model, arch: &str) -> Value {
    let mut v = build_inner(m, arch);
    // Stamped last: it is a hash OF the manifest's compiled-set fields, so it
    // cannot be one of them.
    let h = pairing_hash(&v);
    v["pairing"] = json!({
        "hash": format!("0x{h:016x}"),
        "algo": "fnv1a64 over `union` then `tuning`",
        "note": "A cubin built from this manifest stamps this value as \
                 plow_packet_hash_{lo,hi}; plowrt refuses a module whose stamp \
                 disagrees. A GENERAL object (every arm compiled) carries no stamp \
                 and pairs with any packet.",
    });
    v
}

fn build_inner(m: &Model, arch: &str) -> Value {
    let progs = program_arms(m);
    let union: BTreeSet<Arm> = progs.iter().flat_map(|p| p.arms.iter().cloned()).collect();
    let s = shapes(m);
    let f = features(&union);
    let t = tuning(&s);

    let opcodes: BTreeSet<&str> = union.iter().map(|a| a.op.as_str()).collect();
    let programs: Vec<Value> = progs
        .iter()
        .map(|p| {
            let mut o = Map::new();
            o.insert("kind".into(), json!(p.kind));
            // `bucket` for prefill (chunk rows), `batch` for decode — same field,
            // different meaning, so name it for what it is on each side.
            o.insert(
                if p.kind == "prefill" { "bucket".into() } else { "batch".to_string() },
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
        "opcodes": opcodes,
        "shapes": {
            "hd": s.hd,
            "kv_heads": s.kv_heads,
            "gqa": s.gqa,
            "decode_batch": s.decode_batch,
            "kv_dtype": kv_dtype,
            "max_chunk": s.max_chunk,
            "prefill_buckets": s.prefill_buckets,
        },
        "features": f,
        "tuning": t,
        "programs": programs,
        // What a specialised object must compile: the union over every program
        // and segment. Anything narrower and some bucket hits `default: __trap()`.
        "union": union.iter().map(Arm::key).collect::<Vec<_>>(),
        "analysis": analysis(&progs),
        "backends": { "nvcc": backend_nvcc(&f, &t) },
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
/// Hashed over the `union` and the tuning constants, i.e. exactly what the
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
    out.push_str("/* --- opcodes present in the packet --- */\n");
    for o in DevOp::ALL {
        let present = ops.contains(&op_name(*o));
        let m = o.c_name().replace("PLOW_DOP_", "PLOW_HAS_");
        out.push_str(&format!(
            "#ifndef {m}\n#define {m} {}\n#endif\n",
            if present { 1 } else { 0 }
        ));
    }

    // Head dims the flash family is instantiated at.
    out.push_str("\n/* --- flash head dims present --- */\n");
    for hd in [256u32, 512] {
        let present = union.iter().any(|k| k.ends_with(&format!("/hd{hd}")));
        out.push_str(&format!(
            "#ifndef PLOW_HAS_FLASH_HD{hd}\n#define PLOW_HAS_FLASH_HD{hd} {}\n#endif\n",
            if present { 1 } else { 0 }
        ));
    }

    out.push_str("\n/* --- rule-derived shape constants --- */\n");
    if let Some(t) = manifest.get("tuning").and_then(Value::as_object) {
        if let Some(v) = t.get("gv_mm_max").and_then(Value::as_u64) {
            out.push_str(&format!("#ifndef GV_MM_MAX\n#define GV_MM_MAX {v}\n#endif\n"));
        }
        if let Some(v) = t.get("gf_full").and_then(Value::as_u64) {
            out.push_str(&format!(
                "#ifndef PLOW_NV_FA_GF_FULL\n#define PLOW_NV_FA_GF_FULL {v}\n#endif\n"
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevInst;
    use packet::devbuild::Program;

    fn inst(op: DevOp, i: [u32; 8]) -> DevInst {
        DevInst { op: op as u16, blocks: 1, i, ..Default::default() }
    }

    fn prog(insts: Vec<DevInst>) -> Program {
        Program {
            n_cu: 4, n_counter: 0, insts, stream: vec![], stream_ofs: vec![],
            stream_len: vec![], waits: vec![], succs: vec![], tensors: vec![],
            gq_stream: vec![], gq_seg_ofs: vec![], l2_sms: 0, l2_domains: 0,
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
            n_cu: 170, target: 0, tensors: vec![], progs: vec![pf, dec],
            kv_row_insts: vec![], prog_t: vec![1024, 8], gen: vec![],
        }
    }

    /// The manifest must reflect the STREAM. An op nothing emitted must not
    /// appear just because the emitter could have emitted it.
    #[test]
    fn opcodes_come_from_the_stream() {
        let man = build(&model(), "sm_120a");
        let ops: Vec<&str> = man["opcodes"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
        assert!(ops.contains(&"FlashDecodeFp8"));
        assert!(ops.contains(&"Gemv"));
        assert!(!ops.contains(&"FlashMlaDecode"));
    }

    /// One opcode, two bodies: hd is an instruction field, so hd256 and hd512
    /// must be SEPARATE arms or a specialised object drops one of them.
    #[test]
    fn flash_arms_split_by_head_dim() {
        let man = build(&model(), "sm_120a");
        let union: Vec<&str> = man["union"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
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
            n_cu: 170, target: 0, tensors: vec![], progs: vec![dec],
            kv_row_insts: vec![], prog_t: vec![8], gen: vec![],
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
        let req: Vec<&str> = man["backends"]["nvcc"]["requires"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
        assert!(req.contains(&"PLOW_FP8_KV=1"));
        assert!(req.contains(&"PLOW_NV_PREFILL=1"));
        assert!(!req.contains(&"PLOW_NV_W8A8=1"));
        let rec: Vec<&str> = man["backends"]["nvcc"]["recommends"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
        assert!(rec.contains(&"GV_MM_MAX=8"));
        assert!(rec.contains(&"PLOW_NV_FA_GF_FULL=8"));
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
}
