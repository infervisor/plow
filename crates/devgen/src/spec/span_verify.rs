//! Span verify (`PLOW_EMIT_SPAN_VERIFY`, plowrt exec/gpu/spec.rs): row `r` of a decode rung
//! reads and writes slot `rowslot[r]`, and every recurrent site keeps what a commit needs to
//! leave its buffer after any span row.
//!
//! One table says where each op takes those operands; [`bind`] writes them. Two emitters use it
//! two ways, and the packets differ, so they stay two ways:
//! - inline (`qwen35`): `in.rowslot` and the `[slot][depth]` snapshots of the state after each
//!   non-final span row are declared while the rung is emitted and bound in the op's own closure;
//! - after the fact ([`bind_decode_rungs`], `nemotron_h`): the finished decode rungs are
//!   rewritten, `in.rowslot`, the replay table and per-buffer step records
//!   (`recurrent_state.verify.replay`) appended to the tensor table, plus the replay program
//!   ([`replay_section`]).
//!
//! Folding one into the other moves tensor handles (and snapshot sizes), so it is not a
//! packet-preserving change.
use packet::dev::{DevInst, DevOp, TENSOR_NONE};

use crate::Model;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    T,
    I,
    J,
}

/// An operand position: slot `kind[idx]`, or the `bits`-wide field at `shift` of an integer
/// slot whose low part already carries the op's own immediate.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slot {
    kind: Kind,
    idx: usize,
    shift: u32,
    bits: u32,
}

const fn t(idx: usize) -> Slot {
    Slot {
        kind: Kind::T,
        idx,
        shift: 0,
        bits: 32,
    }
}
const fn i(idx: usize) -> Slot {
    Slot {
        kind: Kind::I,
        idx,
        shift: 0,
        bits: 32,
    }
}
const fn j(idx: usize) -> Slot {
    Slot {
        kind: Kind::J,
        idx,
        shift: 0,
        bits: 32,
    }
}
const fn hi(idx: usize, shift: u32, bits: u32) -> Slot {
    Slot {
        kind: Kind::I,
        idx,
        shift,
        bits,
    }
}

/// The span operands of a recurrent site.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recurrent {
    /// The slot holding the buffer whose snapshot (or record) is bound.
    pub buffer: Slot,
    pub snap: Slot,
    pub depth: Slot,
    /// The replay table, on sites that support replay rollback.
    pub table: Option<Slot>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Rule {
    pub rowslot: Slot,
    pub recurrent: Option<Recurrent>,
}

/// Where `d`'s op takes the span operands, or `None` when the instruction is not a span site
/// (a `QwenHeadNormRope` without a cache is a plain head norm).
pub(crate) fn rule(d: &DevInst) -> Option<Rule> {
    let plain = |rowslot| {
        Some(Rule {
            rowslot,
            recurrent: None,
        })
    };
    let recurrent = |rowslot, buffer, snap, depth, table| {
        Some(Rule {
            rowslot,
            recurrent: Some(Recurrent {
                buffer,
                snap,
                depth,
                table,
            }),
        })
    };
    match DevOp::from_u16(d.op)? {
        DevOp::QwenHeadNormRope if d.i[4] != 0 => plain(t(7)),
        DevOp::HeadNormRopeFp8 => plain(t(7)),
        DevOp::FlashDecode => plain(t(6)),
        DevOp::FlashDecodeFp8 => plain(j(1)),
        DevOp::QwenGdnConv => recurrent(t(5), t(3), t(6), i(3), None),
        DevOp::QwenGdnStep => recurrent(j(1), t(6), i(6), i(7), None),
        // op_mamba_span.cuh: t7 = the conv-input record, i5 = depth | table << 16.
        DevOp::Mamba2Conv => recurrent(t(6), t(4), t(7), hi(5, 0, 16), Some(hi(5, 16, 16))),
        // The operands ride the high halves of the step's own immediates; i7 >> 16 = the record.
        DevOp::Mamba2Step => recurrent(
            hi(6, 16, 16),
            t(6),
            hi(7, 16, 16),
            hi(5, 8, 8),
            Some(hi(5, 16, 16)),
        ),
        _ => None,
    }
}

fn get(d: &DevInst, s: Slot) -> u32 {
    let raw = match s.kind {
        Kind::T => d.t[s.idx],
        Kind::I => d.i[s.idx],
        Kind::J => d.j[s.idx],
    };
    let v = raw >> s.shift;
    if s.bits == 32 {
        v
    } else {
        v & ((1 << s.bits) - 1)
    }
}

fn put(d: &mut DevInst, s: Slot, v: u32) {
    let op = d.op;
    let unbound = if s.kind == Kind::T { TENSOR_NONE } else { 0 };
    assert!(
        get(d, s) == unbound && (s.bits == 32 || v < 1 << s.bits),
        "op {op}: span operand {s:?} already bound or {v} does not fit"
    );
    let slot = match s.kind {
        Kind::T => &mut d.t[s.idx],
        Kind::I => &mut d.i[s.idx],
        Kind::J => &mut d.j[s.idx],
    };
    if s.shift == 0 {
        *slot = v;
    } else {
        *slot |= v << s.shift;
    }
}

/// Binds the span operands of `d`: the row map, and for a recurrent site `(snapshot, depth)`.
pub(crate) fn bind(d: &mut DevInst, rowslot: u32, snap: Option<(u32, u32)>) {
    let r = rule(d).unwrap_or_else(|| panic!("op {} is not a span-verify site", d.op));
    put(d, r.rowslot, rowslot);
    match (r.recurrent, snap) {
        (Some(rc), Some((snap, depth))) => {
            put(d, rc.snap, snap);
            put(d, rc.depth, depth);
        }
        (None, None) => {}
        _ => panic!("op {}: snapshot operands and site kind disagree", d.op),
    }
}

/// The buffer a recurrent site of `d` snapshots.
pub(crate) fn snapshot_buffer(d: &DevInst) -> Option<u32> {
    rule(d)?.recurrent.map(|rc| get(d, rc.buffer))
}

/// Name of the replay table (`recurrent_state.verify.replay.table`).
pub(crate) const REPLAY_TABLE: &str = "in.spec.replay";

/// One record row of a replay site (op_mamba_span.cuh): the conv's inputs (`C` bf16), or the
/// step's x and B (`H * P + G * N` bf16) then dt (`H` f32), padded to 16 bytes.
fn record_row_bytes(d: &DevInst) -> u64 {
    let bytes = match DevOp::from_u16(d.op) {
        Some(DevOp::Mamba2Conv) => u64::from(d.i[0]) * 2,
        Some(DevOp::Mamba2Step) => {
            let (h, p, n, g) = (d.i[0], d.i[1], d.i[2], d.i[3]);
            u64::from(h * p + g * n) * 2 + u64::from(h) * 4
        }
        _ => panic!("op {} has no replay record", d.op),
    };
    bytes.next_multiple_of(16)
}

/// Span verify after the fact over `m`'s decode rungs (replay rollback): appends `in.rowslot`
/// (`slots` rows), the replay table and, per recurrent buffer, a `<buffer>.rec` of two record
/// parities x `slots` launch rows; binds every site. Returns the recorded buffers.
pub(crate) fn bind_decode_rungs(m: &mut Model, slots: u32, depth: u32) -> usize {
    let declare = |m: &mut Model, name: String, bytes: u64| {
        m.tensors.push(packet::devbuild::TensorDecl {
            name,
            bytes,
            init: None,
        });
        m.tensors.len() as u32 - 1
    };
    let rowslot = declare(m, "in.rowslot".into(), u64::from(slots) * 4);
    let table = declare(m, REPLAY_TABLE.into(), (u64::from(slots) + 2) * 4);
    let mut recs: std::collections::BTreeMap<u32, (u32, u64)> = Default::default();
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    for pi in dec_lo..m.progs.len() {
        for ii in 0..m.progs[pi].insts.len() {
            let mut d = m.progs[pi].insts[ii];
            let Some(r) = rule(&d) else {
                continue;
            };
            let rec = snapshot_buffer(&d).map(|buffer| {
                let row = record_row_bytes(&d);
                let &mut (rec, have) = recs.entry(buffer).or_insert_with(|| {
                    let t = &m.tensors[buffer as usize];
                    assert!(
                        t.bytes % u64::from(slots) == 0,
                        "{} is not slot-major",
                        t.name
                    );
                    let name = format!("{}.rec", t.name);
                    (declare(m, name, 2 * u64::from(slots) * row), row)
                });
                assert_eq!(have, row, "one record geometry per buffer");
                (rec, depth)
            });
            bind(&mut d, rowslot, rec);
            if let Some(slot) = r.recurrent.and_then(|rc| rc.table) {
                put(&mut d, slot, table);
            }
            m.progs[pi].insts[ii] = d;
        }
    }
    recs.len()
}

/// The replay program ([`plow_asset::recurrent_state::REPLAY_SECTION`]): the widest decode
/// rung's recurrent sites with no span rows, so each only replays the deferred slots the replay
/// table names. The sites touch disjoint buffers and records: no dependencies.
pub(crate) fn replay_section(m: &Model) -> packet::devbuild::SectionData {
    let widest = m.progs.last().expect("decode rungs");
    let rows = *m.prog_t.last().expect("decode rungs");
    let mut b = packet::devbuild::Builder::new(m.n_cu);
    b.adopt_tensors(m.tensors.clone());
    b.force_uniseg();
    let mut sites = 0;
    for d in &widest.insts {
        let rows_at = match DevOp::from_u16(d.op) {
            Some(DevOp::Mamba2Conv) => 2,
            Some(DevOp::Mamba2Step) => 4,
            _ => continue,
        };
        let op = DevOp::from_u16(d.op).expect("matched");
        let src = *d;
        b.emit(op, b.all(), &[], |e| {
            e.t = src.t;
            e.i = src.i;
            e.f = src.f;
            e.j = src.j;
            e.i[rows_at] = 0;
        });
        sites += 1;
    }
    assert!(sites > 0, "replay program without a recurrent site");
    let p = b.finish();
    let insts: Vec<_> = p.insts.iter().map(|d| d.pack()).collect();
    let view = plow_asset::program::Program {
        rows,
        role: packet::devbuild::ProgramRole::DecodeRung { rows },
        n_counter: p.n_counter,
        insts: &insts,
        stream: &p.stream,
        stream_ofs: &p.stream_ofs,
        stream_len: &p.stream_len,
        waits: &p.waits,
        succs: &p.succs,
        gq_stream: &p.gq_stream,
        gq_seg_ofs: &p.gq_seg_ofs,
        l2_domains: p.l2_domains,
    };
    let data = plow_asset::aux_program::encode(m.n_cu, m.tensors.len(), &[view])
        .unwrap_or_else(|e| panic!("replay program: {e}"));
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: plow_asset::recurrent_state::REPLAY_SECTION.into(),
        data,
    }
}
