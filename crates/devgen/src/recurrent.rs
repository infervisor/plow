//! The RECURRENT STATE section (`plow_asset::recurrent_state`) of a hybrid model: every
//! slot-major state buffer under one name prefix, the active mask, and the verify rollback when
//! span verify gave each buffer a `<buffer>.snap` twin (or a `<buffer>.rec` step record).
use packet::dev::DevOp;
use plow_asset::recurrent_state::{
    Buffer, Manifest, Record, Replay, Snapshot, TensorFacts, VerifyRollback, SECTION, VERSION,
};

use crate::Model;

/// What differs between the recurrent families.
pub(crate) struct Family<'a> {
    /// Name prefix of the state buffers (`state.qwen.`, `state.nemo.`).
    pub prefix: &'a str,
    /// The op whose presence in every prefill program means prefill updates the slot state.
    pub prefill_op: DevOp,
    /// A packet with no state buffer still reports `prefill_updates_state` (vacuously true).
    pub stateless_prefill_updates: bool,
    /// The packed-prefill initstate scratch tensor, when the family has one.
    pub initstate: Option<&'a str>,
    /// The step seed rides `in.active` word `slots` (stochastic state rounding).
    pub step_seed: bool,
    /// Verify rollback replays step records (`<buffer>.rec`, span_verify::bind_decode_rungs).
    pub replay: bool,
}

pub(crate) fn section(m: &Model, slots: u32, f: &Family) -> packet::devbuild::SectionData {
    let buffer = |index: usize| {
        let t = &m.tensors[index];
        assert!(
            t.bytes % u64::from(slots) == 0,
            "{} is not slot-major",
            t.name
        );
        Buffer {
            tensor: u16::try_from(index).expect("tensor handle"),
            slot_bytes: u32::try_from(t.bytes / u64::from(slots)).expect("slot bytes"),
        }
    };
    let position = |name: &str| m.tensors.iter().position(|t| t.name == name);
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    let buffers: Vec<Buffer> = (0..m.tensors.len())
        .filter(|&i| {
            let name = &m.tensors[i].name;
            name.starts_with(f.prefix) && !name.ends_with(".snap") && !name.ends_with(".rec")
        })
        .map(buffer)
        .collect();
    let snapshots: Vec<Snapshot> = buffers
        .iter()
        .filter_map(|b| {
            position(&format!("{}.snap", m.tensors[b.tensor as usize].name)).map(|t| Snapshot {
                buffer: b.tensor,
                tensor: u16::try_from(t).expect("tensor handle"),
            })
        })
        .collect();
    let records: Vec<Record> = buffers
        .iter()
        .filter_map(|b| {
            let t = position(&format!("{}.rec", m.tensors[b.tensor as usize].name))?;
            Some(Record {
                buffer: b.tensor,
                tensor: u16::try_from(t).expect("tensor handle"),
                row_bytes: u32::try_from(m.tensors[t].bytes / (2 * u64::from(slots)))
                    .expect("record row bytes"),
            })
        })
        .collect();
    assert!(
        f.replay || records.is_empty(),
        "step records in a family without replay rollback"
    );
    let kept = if f.replay {
        records.len()
    } else {
        snapshots.len()
    };
    let verify = (kept > 0).then(|| {
        assert_eq!(
            kept,
            buffers.len(),
            "every recurrent buffer needs a verify snapshot or record"
        );
        VerifyRollback {
            depth: crate::spec::verify_depth(),
            snapshots,
            replay: f.replay.then(|| Replay {
                table: u16::try_from(
                    position(crate::spec::span_verify::REPLAY_TABLE).expect("replay table"),
                )
                .expect("tensor handle"),
                records,
            }),
        }
    });
    let manifest = Manifest {
        version: VERSION,
        slots,
        prefill_updates_state: (f.stateless_prefill_updates && buffers.is_empty())
            || m.progs[..dec_lo]
                .iter()
                .all(|p| p.insts.iter().any(|i| i.op == f.prefill_op as u16)),
        buffers,
        active: u16::try_from(position("in.active").expect("in.active")).expect("tensor handle"),
        prefill_initstate: f.initstate.and_then(position).map(buffer),
        step_seed: f.step_seed,
        verify,
    };
    manifest
        .validate(slots, |h| {
            m.tensors.get(h as usize).map(|t| TensorFacts {
                bytes: t.bytes,
                initialized: t.init.is_some(),
            })
        })
        .unwrap_or_else(|error| panic!("recurrent state: {error}"));
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&manifest).expect("recurrent state"),
    }
}
