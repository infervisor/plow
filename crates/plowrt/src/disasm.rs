//! Rendering for `plowrt disasm` — text and JSON over a parsed device blob.
//!
//! The structural half lives in [`packet::disasm`], which has no dependencies
//! and therefore cannot serialize. This module owns the serde types, the
//! kernarg/dispatch summary, and the text form.
//!
//! # Size
//!
//! Measured on a real GLM-5.2 TP8 blob: a prefill program is 2021 instructions
//! but **377,444 stream entries**. Instruction-level output is under a megabyte;
//! stream entries are ~45 MB per program. So instructions are the default and
//! stream entries are opt-in, and `jsonl` exists so that a large dump can be
//! piped through `jq` a record at a time instead of parsed whole.

use std::collections::BTreeMap;

use packet::disasm::{disasm, int_slot_label, op_name, Inst};
use packet::slots::Provenance;
use serde::Serialize;

use crate::analysis::{counters, graph};
use crate::asset::devblob::{DevBlob, DevProg};

/// What to include. Every section past the instructions costs real time or
/// bytes, so none of them is on by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sections {
    pub kernargs: bool,
    pub tensors: bool,
    pub counters: bool,
    pub stream: bool,
    /// Skip every derived metric — structure only.
    pub no_analysis: bool,
}

#[derive(Serialize)]
pub struct BlobReport<'a> {
    pub blob: String,
    pub n_cu: u32,
    pub flags: u32,
    /// GPU fingerprint the blob was compiled for; 0 = unknown.
    pub target: u32,
    pub tp: Option<TpJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<Vec<TensorJson<'a>>>,
    pub programs: Vec<ProgramReport<'a>>,
}

#[derive(Serialize)]
pub struct TpJson {
    pub n_gpu: u32,
    pub hidden: u32,
    pub slot_bytes: u64,
}

#[derive(Serialize)]
pub struct TensorJson<'a> {
    pub handle: usize,
    pub name: &'a str,
    pub bytes: u64,
    /// `WEIGHT` / `TABLE` / `RUNTIME`, from [`packet::names`] — see `class_of`.
    pub class: &'static str,
    /// Carries bytes in the blob's init section.
    pub init: bool,
}

#[derive(Serialize)]
pub struct ProgramReport<'a> {
    /// The T this program was compiled for; decode is 1.
    pub t: u32,
    pub n_inst: usize,
    pub n_counter: u32,
    pub n_stream_ent: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernargs: Option<KernargJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counters: Option<CounterReportJson>,
    pub insts: Vec<InstJson<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<Vec<StreamEntJson>>,
}

/// The `DevProgram` kernarg block, as far as a file can describe it.
///
/// Every pointer field is a **device address** filled in by the host at launch,
/// so a static dump cannot show values — but *which* fields the program needs
/// populated is the dispatch configuration, and that is recoverable. Fields the
/// host owns are reported as `"runtime"` rather than a fabricated zero.
#[derive(Serialize)]
pub struct KernargJson {
    pub size_bytes: usize,
    pub pointers: BTreeMap<&'static str, &'static str>,
    pub scalars: BTreeMap<&'static str, String>,
    pub derived: DerivedJson,
}

#[derive(Serialize)]
pub struct DerivedJson {
    /// `global-queue` when the blob carries a `gq_stream`, else `static`.
    pub scheduler: &'static str,
    pub segmented: bool,
    pub n_seg: usize,
    pub l2_placed: bool,
    pub tensor_parallel: bool,
    pub n_gpu: u32,
}

#[derive(Serialize)]
pub struct InstJson<'a> {
    pub idx: usize,
    pub op: Option<u16>,
    pub op_name: Option<&'static str>,
    pub blocks: u16,
    /// `documented` / `inherited:<Op>` / `reserved` / `undocumented`.
    pub slots: String,
    pub tensors: Vec<TensorOperandJson<'a>>,
    pub ints: Vec<IntOperandJson>,
    pub floats: Vec<FloatOperandJson>,
    /// The wire bytes. Always present — see [`packet::disasm`].
    pub raw: RawJson,
}

#[derive(Serialize)]
pub struct TensorOperandJson<'a> {
    pub slot: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'static str>,
    /// `None` when the slot holds the absent-operand sentinel.
    pub handle: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor: Option<&'a str>,
    pub present: bool,
    pub optional: bool,
}

#[derive(Serialize)]
pub struct IntOperandJson {
    pub slot: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'static str>,
    pub value: u32,
}

#[derive(Serialize)]
pub struct FloatOperandJson {
    pub slot: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'static str>,
    pub value: f32,
}

#[derive(Serialize)]
pub struct RawJson {
    pub op: u16,
    pub blocks: u16,
    pub t: [u16; 8],
    pub i: [u32; 8],
    /// Hex, because two thirds of it is an overlay and decimal invites reading
    /// a float bit pattern as a count.
    pub fj: [String; 3],
}

#[derive(Serialize)]
pub struct StreamEntJson {
    pub inst: u32,
    pub slice: u32,
    pub wait_ofs: u32,
    pub wait_len: u16,
    pub succ_ofs: u32,
    pub succ_len: u16,
    pub flags: u16,
    pub seg: u16,
}

/// graphstat's aggregates, plus the per-counter detail layered above them.
#[derive(Serialize)]
pub struct CounterReportJson {
    pub aggregate: AggregateJson,
    pub per_counter: Vec<CounterJson>,
    pub dead: Vec<CounterJson>,
    pub redundant_edges: Vec<RedundantEdgeJson>,
    pub threshold_histogram: BTreeMap<u32, u32>,
    pub liveness: LivenessJson,
}

/// Exactly what `graphstat` prints, from the same code.
#[derive(Serialize)]
pub struct AggregateJson {
    pub ops: usize,
    pub counters: u32,
    pub se_fine: usize,
    pub edges: usize,
    pub edges_tr: usize,
    pub redundant: usize,
    pub dead: usize,
    pub polls: u64,
    pub polls_tr: u64,
    pub polls_removable: u64,
    pub bumps: u64,
    pub bumps_live: u64,
    pub critical_path: u32,
}

#[derive(Serialize, Clone)]
pub struct CounterJson {
    pub id: u32,
    pub threshold: Option<u32>,
    pub producer: Option<u32>,
    pub producer_op: Option<&'static str>,
    pub consumers: Vec<u32>,
    pub fan_out: usize,
    pub dead: bool,
    pub poll_cost: u64,
    pub bump_cost: u64,
}

#[derive(Serialize)]
pub struct RedundantEdgeJson {
    pub from: u32,
    pub to: u32,
    pub via: Vec<u32>,
    pub polls_saved: u64,
    pub co_placed: bool,
}

#[derive(Serialize)]
pub struct LivenessJson {
    pub max_concurrent: u32,
    pub at_inst: u32,
    pub p50: u32,
    pub p99: u32,
}

fn provenance_str(p: Provenance) -> String {
    match p {
        Provenance::Documented => "documented".into(),
        Provenance::Inherited(base) => format!("inherited:{}", op_name(base)),
        Provenance::Reserved => "reserved".into(),
        Provenance::Undocumented => "undocumented".into(),
    }
}

/// Classify a declared tensor by who fills it.
///
/// The three [`packet::names`] predicates PARTITION the namespace —
/// `is_checkpoint_weight` is defined as "neither runtime nor host-filled" — so
/// these three labels are exhaustive and a fourth catch-all would be
/// unreachable. Reusing the predicates rather than re-deriving them is the point:
/// a second classifier that disagreed with the loader's would mislabel exactly
/// the tensors a reader is checking.
fn class_of(name: &str) -> &'static str {
    if packet::names::is_host_filled_table(name) {
        "TABLE"
    } else if packet::names::is_runtime_tensor(name) {
        "RUNTIME"
    } else {
        "WEIGHT"
    }
}

fn inst_json<'a>(d: &Inst<'a>) -> InstJson<'a> {
    InstJson {
        idx: d.idx,
        op: d.op.map(|o| o as u16),
        op_name: d.op_name,
        blocks: d.blocks,
        slots: provenance_str(d.provenance),
        tensors: d
            .tensors
            .iter()
            .map(|o| TensorOperandJson {
                slot: format!("t{}", o.slot),
                name: o.name,
                handle: o.handle,
                tensor: o.tensor,
                present: o.handle.is_some(),
                optional: o.optional,
            })
            .collect(),
        ints: d
            .ints
            .iter()
            .map(|o| IntOperandJson {
                slot: int_slot_label(o.slot),
                name: o.name,
                value: o.value,
            })
            .collect(),
        floats: d
            .floats
            .iter()
            .map(|o| FloatOperandJson {
                slot: format!("f{}", o.slot),
                name: o.name,
                value: o.value,
            })
            .collect(),
        raw: RawJson {
            op: d.raw.op,
            blocks: d.raw.blocks,
            t: d.raw.t,
            i: d.raw.i,
            fj: [
                format!("0x{:08x}", d.raw.fj[0]),
                format!("0x{:08x}", d.raw.fj[1]),
                format!("0x{:08x}", d.raw.fj[2]),
            ],
        },
    }
}

fn kernargs(blob: &DevBlob, prog: &DevProg) -> KernargJson {
    let mut pointers: BTreeMap<&'static str, &'static str> = BTreeMap::new();
    let mark = |populated: bool| if populated { "populated" } else { "null" };
    let tp = blob.tp.is_some();
    // Always built by the host from blob content.
    for k in [
        "insts",
        "stream",
        "stream_ofs",
        "stream_len",
        "waits",
        "succs",
        "counters",
        "tensors",
    ] {
        pointers.insert(k, mark(true));
    }
    pointers.insert("gq_stream", mark(!prog.gq_stream.is_empty()));
    pointers.insert("gq_seg_ofs", mark(!prog.gq_seg_ofs.is_empty()));
    pointers.insert("gq_cursor", mark(!prog.gq_stream.is_empty()));
    pointers.insert("trace", "runtime (PLOW_TRACE_RAW)");
    pointers.insert("seg_ofs", "runtime (static_seg_ofs at load)");
    pointers.insert("xctr", mark(tp));
    pointers.insert("peer_scratch", mark(tp));

    let n_seg = prog.gq_seg_ofs.len().saturating_sub(1);
    let mut scalars: BTreeMap<&'static str, String> = BTreeMap::new();
    scalars.insert("cur_seg", "runtime (one launch per segment)".into());
    scalars.insert("l2_domains", prog.l2_domains.to_string());
    scalars.insert("n_seg", n_seg.to_string());
    scalars.insert("rank", "runtime (host fills per rank)".into());
    scalars.insert(
        "n_gpu",
        blob.tp
            .as_ref()
            .map(|t| t.n_gpu.to_string())
            .unwrap_or_else(|| "1".into()),
    );

    KernargJson {
        size_bytes: std::mem::size_of::<packet::dev::DevProgram>(),
        pointers,
        scalars,
        derived: DerivedJson {
            scheduler: if prog.gq_stream.is_empty() {
                "static"
            } else {
                "global-queue"
            },
            segmented: n_seg > 1,
            n_seg,
            l2_placed: prog.l2_domains > 0,
            tensor_parallel: tp,
            n_gpu: blob.tp.as_ref().map(|t| t.n_gpu).unwrap_or(1),
        },
    }
}

fn counter_json(c: &counters::Counter, insts: &[packet::dev::DevInst64]) -> CounterJson {
    CounterJson {
        id: c.id,
        threshold: c.threshold,
        producer: c.producer,
        producer_op: c
            .producer
            .and_then(|p| insts.get(p as usize))
            .and_then(|w| packet::dev::DevOp::from_u16(w.op))
            .map(op_name),
        fan_out: c.consumers.len(),
        consumers: c.consumers.clone(),
        dead: c.dead,
        poll_cost: c.poll_cost,
        bump_cost: c.bump_cost,
    }
}

fn counters_json(prog: &DevProg, s: &graph::Stats) -> CounterReportJson {
    let r = counters::Report::of(prog, s);
    let (depth, _) = graph::critical_path(s.n_ops, &s.edges);
    CounterReportJson {
        aggregate: AggregateJson {
            ops: s.n_ops,
            counters: s.n_counter,
            se_fine: s.fine_ents,
            edges: s.edges.len(),
            edges_tr: s.edges_tr.len(),
            redundant: s.edges.len() - s.edges_tr.len(),
            dead: s.dead_ctr.len(),
            polls: s.polls,
            polls_tr: s.polls_tr,
            polls_removable: s.polls.saturating_sub(s.polls_tr),
            bumps: s.bumps,
            bumps_live: s.bumps_live,
            critical_path: depth,
        },
        per_counter: r
            .counters
            .iter()
            .map(|c| counter_json(c, &prog.insts))
            .collect(),
        dead: r
            .dead()
            .into_iter()
            .map(|c| counter_json(c, &prog.insts))
            .collect(),
        redundant_edges: r
            .redundant
            .iter()
            .map(|e| RedundantEdgeJson {
                from: e.from,
                to: e.to,
                via: e.via.clone(),
                polls_saved: e.polls_saved,
                co_placed: e.co_placed,
            })
            .collect(),
        threshold_histogram: r.threshold_histogram.clone(),
        liveness: LivenessJson {
            max_concurrent: r.liveness.max_concurrent,
            at_inst: r.liveness.at_inst,
            p50: r.liveness.p50,
            p99: r.liveness.p99,
        },
    }
}

/// Build the report. `range` restricts the instruction window; `only_t`
/// restricts to one program.
pub fn report<'a>(
    blob: &'a DevBlob,
    path: &str,
    sec: Sections,
    only_t: Option<u32>,
    range: Option<(usize, usize)>,
) -> BlobReport<'a> {
    let names: Vec<&str> = blob.tensors.iter().map(|t| t.name.as_str()).collect();

    let programs = blob
        .progs
        .iter()
        .filter(|p| only_t.is_none_or(|t| p.t == t))
        .map(|p| {
            let (lo, hi) = range.unwrap_or((0, p.insts.len()));
            let (lo, hi) = (lo.min(p.insts.len()), hi.min(p.insts.len()));
            let stats = (!sec.no_analysis && sec.counters).then(|| graph::analyse(p));
            ProgramReport {
                t: p.t,
                n_inst: p.insts.len(),
                n_counter: p.n_counter,
                n_stream_ent: p.stream.len(),
                kernargs: sec.kernargs.then(|| kernargs(blob, p)),
                counters: stats.as_ref().map(|s| counters_json(p, s)),
                insts: p.insts[lo..hi]
                    .iter()
                    .enumerate()
                    .map(|(k, w)| inst_json(&disasm(lo + k, w, &names)))
                    .collect(),
                stream: sec.stream.then(|| {
                    p.stream
                        .iter()
                        .map(|e| StreamEntJson {
                            inst: e.inst,
                            slice: e.slice,
                            wait_ofs: e.wait_ofs,
                            wait_len: e.wait_len,
                            succ_ofs: e.succ_ofs,
                            succ_len: e.succ_len,
                            flags: e.flags,
                            seg: e.seg,
                        })
                        .collect()
                }),
            }
        })
        .collect();

    BlobReport {
        blob: path.to_string(),
        n_cu: blob.n_cu,
        flags: blob.flags,
        target: blob.target,
        tp: blob.tp.as_ref().map(|t| TpJson {
            n_gpu: t.n_gpu,
            hidden: t.hidden,
            slot_bytes: t.slot_bytes,
        }),
        tensors: sec.tensors.then(|| {
            blob.tensors
                .iter()
                .enumerate()
                .map(|(h, t)| TensorJson {
                    handle: h,
                    name: &t.name,
                    bytes: t.bytes,
                    class: class_of(&t.name),
                    init: t.init.is_some(),
                })
                .collect()
        }),
        programs,
    }
}

/// One instruction as a single greppable line.
///
/// Compact rather than a block per operand: a program is thousands of
/// instructions, and the common use is `grep`/`less` over the whole thing.
pub fn text_inst(d: &Inst<'_>) -> String {
    let mut s = format!(
        "#{:<5} {:<24} b={:<4}",
        d.idx,
        d.op_name
            .map(str::to_string)
            .unwrap_or_else(|| format!("op{}?", d.raw.op)),
        d.blocks
    );
    for o in &d.tensors {
        let label = o.name.unwrap_or("?");
        match (o.handle, o.tensor) {
            (None, _) => s.push_str(&format!(" {label}<-—")),
            (Some(h), None) => s.push_str(&format!(" {label}<-#{h}")),
            (Some(_), Some(t)) => s.push_str(&format!(" {label}<-{t}")),
        }
    }
    if !d.ints.is_empty() || !d.floats.is_empty() {
        s.push_str(" |");
    }
    for o in &d.ints {
        s.push_str(&format!(
            " {}={}",
            o.name.unwrap_or(int_slot_label(o.slot)),
            o.value
        ));
    }
    for o in &d.floats {
        s.push_str(&format!(" {}={}", o.name.unwrap_or("f"), o.value));
    }
    if d.provenance == Provenance::Undocumented {
        s.push_str("   [no slot spec — raw]");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three labels partition the namespace, so every declared tensor gets
    /// exactly one — including the `fp8/` twins and the untied head, which carry
    /// no `model.` prefix.
    #[test]
    fn tensor_classes_use_the_shared_predicates() {
        assert_eq!(
            class_of("model.layers.0.self_attn.q_a_proj.weight"),
            "WEIGHT"
        );
        assert_eq!(class_of("lm_head.weight"), "WEIGHT");
        assert_eq!(class_of("act.x"), "RUNTIME");
        assert_eq!(class_of("kv.0.ckv"), "RUNTIME");
        assert_eq!(class_of("model.layers.3.mlp.expert_weight_table"), "TABLE");
    }

    #[test]
    fn provenance_renders_its_base() {
        assert_eq!(
            provenance_str(Provenance::Inherited(packet::dev::DevOp::Gemm)),
            "inherited:Gemm"
        );
        assert_eq!(provenance_str(Provenance::Documented), "documented");
    }

    /// The overlay is hex in JSON: `fj[1]` may be a float bit pattern, and
    /// decimal invites reading one as a count.
    #[test]
    fn raw_fj_is_hex() {
        let mut d = packet::dev::DevInst {
            op: packet::dev::DevOp::Residual as u16,
            ..Default::default()
        };
        d.f[0] = 1.0;
        let j = inst_json(&disasm(0, &d.pack(), &[]));
        assert_eq!(j.raw.fj[0], "0x3f800000");
    }
}

/// The scheduled `.pkt` form (`plowc --emit packets`) — a second, much simpler
/// backend behind the same command.
///
/// It needs no slot table: [`packet::Body`] is already an enum of named fields,
/// so the field names *are* the operand names. Counters are explicit here too
/// ([`packet::Counter`] carries `scope`), where the device blob leaves scope to
/// be inferred.
pub mod sched {
    use packet::{Body, Opcode, Program};
    use serde::Serialize;
    use serde_json::{json, Value};

    #[derive(Serialize)]
    pub struct SchedReport {
        pub file: String,
        pub bucket_id: u16,
        pub plan_gen: u16,
        pub flags: u16,
        pub n_inst: usize,
        pub n_counter: usize,
        pub counters: Vec<CounterJson>,
        pub insts: Vec<InstJson>,
    }

    #[derive(Serialize)]
    pub struct CounterJson {
        pub id: u32,
        pub threshold: u32,
        /// `intra_sm` / `intra_gpu` / `cross_unit` — recorded in the stream, not
        /// inferred, which is the one thing this format gives that the device
        /// blob does not.
        pub scope: &'static str,
    }

    #[derive(Serialize)]
    pub struct InstJson {
        pub idx: usize,
        pub opcode: u16,
        pub opcode_name: String,
        pub family: &'static str,
        pub variant: u8,
        pub resource: String,
        pub unit: u8,
        pub index: u16,
        pub body: Value,
        pub wait: Vec<u32>,
        pub succ: Vec<u32>,
    }

    fn scope_name(s: u8) -> &'static str {
        match s {
            0 => "intra_sm",
            1 => "intra_gpu",
            2 => "cross_unit",
            _ => "?",
        }
    }

    fn family_name(f: u8) -> &'static str {
        match f {
            0 => "CONTROL",
            1 => "DMA",
            2 => "RDMA",
            3 => "GEMM",
            4 => "FLASH",
            5 => "ROW",
            6 => "LAYOUT",
            7 => "TOKEN",
            _ => "?",
        }
    }

    fn opcode_name(op: Opcode) -> String {
        match op {
            Opcode::NOP => "NOP".into(),
            Opcode::HOST_COORD => "HOST_COORD".into(),
            Opcode::TMA_LOAD => "TMA_LOAD".into(),
            Opcode::TMA_STORE => "TMA_STORE".into(),
            Opcode::RDMA => "RDMA".into(),
            Opcode::SAMPLE => "SAMPLE".into(),
            Opcode::TOKENIZE => "TOKENIZE".into(),
            Opcode::SAMPLE_BATCH => "SAMPLE_BATCH".into(),
            // The families below are variant-bearing, so the bare constant only
            // names variant 0; render `FAMILY/v<n>` rather than mislabel a
            // variant as its golden form.
            o if o.variant() == 0 => family_name(o.family()).to_string(),
            o => format!("{}/v{}", family_name(o.family()), o.variant()),
        }
    }

    /// Body fields verbatim. Names come from the enum, so there is no table to
    /// drift — the reason this backend needs none of `packet::slots`.
    fn body_json(b: &Body) -> Value {
        match b {
            Body::Dma {
                load,
                bytes,
                slot,
                tensor,
                kind,
                access,
            } => json!({
                "load": load, "bytes": bytes, "slot": slot,
                "tensor": tensor, "kind": kind, "access": access
            }),
            Body::Rdma {
                bytes,
                src_unit,
                dst_unit,
            } => {
                json!({ "bytes": bytes, "src_unit": src_unit, "dst_unit": dst_unit })
            }
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
                variant,
            } => json!({
                "coord": coord, "m": m, "n": n, "k": k,
                "tile": { "bm": bm, "bn": bn, "bk": bk },
                "out": out, "tmem": tmem, "variant": variant
            }),
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
                variant,
            } => {
                json!({
                    "coord": coord, "seq_q": seq_q, "seq_kv": seq_kv, "head_dim": head_dim,
                    "tile": { "bq": bq, "bkv": bkv }, "heads": heads,
                    "out": out, "tmem": tmem, "variant": variant
                })
            }
            Body::Row {
                reduce,
                coord,
                rows,
                feat,
                operands,
                br,
                out,
                variant,
            } => json!({
                "reduce": reduce, "coord": coord, "rows": rows, "feat": feat,
                "operands": operands, "br": br, "out": out, "variant": variant
            }),
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
            } => json!({
                "kind": kind, "rank": rank, "elem_size": elem_size, "out": out,
                "shape": shape, "in_stride": in_stride, "out_stride": out_stride,
                "in_base": in_base, "out_base": out_base
            }),
            Body::Token {
                in_slot,
                out_slot,
                kind,
                vocab,
                arg,
            } => json!({
                "in_slot": in_slot, "out_slot": out_slot,
                "kind": kind, "vocab": vocab, "arg": arg
            }),
            Body::Host => json!({}),
        }
    }

    pub fn report(p: &Program, file: &str) -> SchedReport {
        SchedReport {
            file: file.to_string(),
            bucket_id: p.bucket_id,
            plan_gen: p.plan_gen,
            flags: p.flags,
            n_inst: p.insts.len(),
            n_counter: p.counters.len(),
            counters: p
                .counters
                .iter()
                .map(|c| CounterJson {
                    id: c.id,
                    threshold: c.threshold,
                    scope: scope_name(c.scope),
                })
                .collect(),
            insts: p
                .insts
                .iter()
                .enumerate()
                .map(|(idx, i)| {
                    let op = i.body.opcode();
                    InstJson {
                        idx,
                        opcode: op.0,
                        opcode_name: opcode_name(op),
                        family: family_name(op.family()),
                        variant: op.variant(),
                        resource: format!("{:?}", i.resource),
                        unit: i.unit,
                        index: i.index,
                        body: body_json(&i.body),
                        wait: i.wait.clone(),
                        succ: i.succ.clone(),
                    }
                })
                .collect(),
        }
    }

    pub fn text(r: &SchedReport) -> String {
        let mut s = format!(
            "file      {}\nbucket    {}  plan_gen={} flags=0x{:x}\ninsts     {}\ncounters  {}\n",
            r.file, r.bucket_id, r.plan_gen, r.flags, r.n_inst, r.n_counter
        );
        for i in &r.insts {
            s.push_str(&format!(
                "#{:<5} {:<14} {}{:<3} idx={:<5} {} wait{:?} -> succ{:?}\n",
                i.idx, i.opcode_name, i.resource, i.unit, i.index, i.body, i.wait, i.succ
            ));
        }
        s
    }
}

#[cfg(test)]
mod sched_tests {
    use super::sched;
    use packet::{Body, Counter, Inst, Program, ResourceKind};

    fn prog() -> Program {
        Program {
            insts: vec![
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 7,
                    body: Body::Gemm {
                        coord: [0, 1],
                        m: 1,
                        n: 6144,
                        k: 2048,
                        bm: 64,
                        bn: 128,
                        bk: 64,
                        out: 3,
                        tmem: 0xFFFF,
                        variant: 0,
                    },
                    wait: vec![1, 2],
                    succ: vec![2],
                },
                Inst {
                    resource: ResourceKind::Dma,
                    unit: 0,
                    index: 8,
                    body: Body::Dma {
                        load: true,
                        bytes: 256,
                        slot: 1,
                        tensor: 4,
                        kind: 0,
                        access: 0,
                    },
                    wait: vec![],
                    succ: vec![1],
                },
            ],
            counters: vec![
                Counter {
                    id: 1,
                    threshold: 4,
                    scope: 0,
                    _pad: [0; 3],
                },
                Counter {
                    id: 2,
                    threshold: 8,
                    scope: 2,
                    _pad: [0; 3],
                },
            ],
            bucket_id: 5,
            plan_gen: 2,
            flags: 0,
        }
    }

    /// Body field names come straight from the enum, so the JSON is the schedule
    /// without an interpretation layer in between.
    #[test]
    fn body_fields_render_verbatim() {
        let r = sched::report(&prog(), "t.pkt");
        let g = &r.insts[0];
        assert_eq!(g.family, "GEMM");
        assert_eq!(g.resource, "Sm");
        assert_eq!(g.body["m"], 1);
        assert_eq!(g.body["n"], 6144);
        assert_eq!(g.body["tile"]["bm"], 64);
        assert_eq!(g.wait, vec![1, 2]);
        assert_eq!(g.succ, vec![2]);

        let d = &r.insts[1];
        assert_eq!(d.opcode_name, "TMA_LOAD");
        assert_eq!(d.body["bytes"], 256);
        assert!(d.body["load"].as_bool().unwrap());
    }

    /// Counter scope is RECORDED in this format. The device blob leaves it to be
    /// inferred, so this is the one thing the scheduled form says outright.
    #[test]
    fn counter_scope_is_read_not_inferred() {
        let r = sched::report(&prog(), "t.pkt");
        assert_eq!(r.counters[0].scope, "intra_sm");
        assert_eq!(r.counters[1].scope, "cross_unit");
        assert_eq!(r.counters[1].threshold, 8);
    }

    /// A variant-bearing family must not be labelled as its golden form: opcode
    /// `0x0301` is a bf16 GEMM, not `GEMM`.
    #[test]
    fn variants_are_not_labelled_as_golden() {
        let mut p = prog();
        let Body::Gemm {
            ref mut variant, ..
        } = p.insts[0].body
        else {
            unreachable!()
        };
        *variant = packet::Opcode::VARIANT_BF16;
        let r = sched::report(&p, "t.pkt");
        assert_eq!(
            r.insts[0].opcode_name,
            format!("GEMM/v{}", packet::Opcode::VARIANT_BF16)
        );
        assert_ne!(r.insts[0].opcode_name, "GEMM");
    }

    /// What the tool actually reads is a file, so the path that matters is
    /// `decode`d bytes rather than an in-memory `Program`.
    #[test]
    fn survives_a_wire_round_trip() {
        let p = prog();
        let decoded = Program::decode(&p.to_bytes()).expect("decode");
        let r = sched::report(&decoded, "t.pkt");
        assert_eq!(r.n_inst, 2);
        assert_eq!(r.n_counter, 2);
        assert_eq!(r.bucket_id, 5);
        assert_eq!(r.plan_gen, 2);
        assert_eq!(r.insts[0].body["k"], 2048);
    }
}
