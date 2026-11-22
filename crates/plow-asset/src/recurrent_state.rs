//! RECURRENT STATE: the per-slot state tensors of a packet whose layers carry state across
//! tokens (gated delta nets, convolutions, ...) and how the runtime drives them.
//!
//! Every buffer is `[slots][slot_bytes]` slot-major: slot `b` of tensor `t` lives at
//! `base + b * slot_bytes`, `slots` being the packet's physical slot capacity (its widest decode
//! rung). The runtime zeroes every buffer's slot on admission, keeps it across a request's steps
//! and restores it only from a snapshot it took at the slot's exact position (the bench
//! rewind); serving never rewinds it. `active` is `i32[slots]`, one entry per physical slot,
//! `> 0` where the slot takes part in the launch; the runtime fills it before every decode step
//! and prompt launch. `buffers` may be empty for a packet that only takes the active mask.
//! `step_seed` appends one word to `active` (`i32[slots + 1]`): a fresh u32 the runtime writes
//! with the mask before every launch, the per-step seed of stochastic state rounding.
//! `prefill_initstate`, when present, is the gathered initial-state scratch of a packed external
//! prompt scan (`[slots][slot_bytes]`). `prefill_updates_state` says every prefill program
//! advances every buffer for every prompt token it consumes; without it a prompt is consumed by
//! the decode program. A packet whose tensor table has a `state.` class tensor or the active mask
//! input carries this section, and every `state.` class tensor is one of its buffers; a packet
//! without it, or with a `state.` class tensor it does not declare, is refused.
//!
//! `verify` ([`VerifyRollback`]) declares speculative-verify rollback: a span-verify decode launch
//! that runs `n <= depth + 1` consecutive rows of one slot (rows mapped through `in.rowslot`)
//! writes every buffer's state after span row `j < n - 1` into its snapshot tensor at
//! `[slot][j]` and the state after the last row into the buffer itself. Committing the span's
//! first `c` rows (`1 <= c <= n`) leaves the buffer as it is when `c == n` and otherwise copies
//! snapshot `[slot][c - 1]` over the buffer's slot. Snapshot tensors are `state.` class,
//! `[slots][depth][slot_bytes]`, never zeroed (a verify writes a row before any commit reads it).
//! With `replay` ([`Replay`]) a verify keeps no snapshot: it leaves every buffer at the state
//! before the span and records each span row's step inputs, and a commit defers replaying the
//! committed records onto the buffer (`snapshots` is empty).
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "recurrent_state";
pub const VERSION: u32 = 1;
/// The replay program ([`Replay`]): one decode-interpreter program, an `aux_program` section.
pub const REPLAY_SECTION: &str = "recurrent_state.replay";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub tensor: u16,
    pub slot_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub slots: u32,
    pub buffers: Vec<Buffer>,
    pub active: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_initstate: Option<Buffer>,
    pub prefill_updates_state: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub step_seed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<VerifyRollback>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyRollback {
    /// Snapshot rows per slot: the longest committable span is `depth + 1` rows.
    pub depth: u32,
    pub snapshots: Vec<Snapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<Replay>,
}

/// Replay rollback. A span-verify launch leaves every buffer at the state before its spans and
/// writes span row `j`'s step inputs into each buffer's record (`[2][slots][row_bytes]`: record
/// parity, launch row). Committing `c` rows of a span defers replaying its first `c` records onto
/// the buffer: the next row-slot launch (a verify, or the [`REPLAY_SECTION`] program that any
/// other reader of the state runs first) replays every deferred slot before anything else reads
/// it. Both read the replay table `table`, `i32[slots + 2]` written before each such launch:
/// `[0]` = the record parity the launch writes, `[1]` = `slots`, `[2 + s]` = slot `s`'s
/// [`Replay::entry`] (0 = nothing deferred).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    pub table: u16,
    pub records: Vec<Record>,
}

/// The step-input record of one declared buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub buffer: u16,
    pub tensor: u16,
    pub row_bytes: u32,
}

impl Replay {
    /// A replay-table slot entry: replay `rows` records from launch row `row0` of record parity
    /// `parity`; `here` = the slot runs a span in this launch (its span replays first).
    pub fn entry(rows: u32, row0: u32, parity: u32, here: bool) -> Result<u32, String> {
        if rows == 0 || rows > 0xFF || row0 > 0xFF || parity > 1 {
            return Err(format!(
                "replay entry: {rows} rows at row {row0}, parity {parity}"
            ));
        }
        Ok(rows | row0 << 8 | parity << 16 | u32::from(here) << 17)
    }
}

/// The snapshot tensor of one declared buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub buffer: u16,
    pub tensor: u16,
}

/// One device copy a commit performs: `bytes` from `src` to `dst` (byte offsets into the
/// snapshot and buffer tensors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitCopy {
    pub buffer: u16,
    pub snapshot: u16,
    pub dst: u64,
    pub src: u64,
    pub bytes: u64,
}

impl VerifyRollback {
    /// The copies that commit `committed` of a verified span's `rows` rows of slot `slot`
    /// (`buffers` = the manifest's buffers). Empty when the whole span is kept, and always with
    /// `replay` (the runtime defers the replay instead).
    pub fn commit(
        &self,
        buffers: &[Buffer],
        slot: u32,
        rows: u32,
        committed: u32,
    ) -> Result<Vec<CommitCopy>, String> {
        if rows == 0 || rows > self.depth + 1 || committed == 0 || committed > rows {
            return Err(format!(
                "recurrent verify commit: {committed} of {rows} rows (depth {})",
                self.depth
            ));
        }
        if committed == rows || self.replay.is_some() {
            return Ok(Vec::new());
        }
        self.snapshots
            .iter()
            .map(|s| {
                let b = buffers
                    .iter()
                    .find(|b| b.tensor == s.buffer)
                    .ok_or_else(|| format!("snapshot of undeclared buffer {}", s.buffer))?;
                let bytes = u64::from(b.slot_bytes);
                Ok(CommitCopy {
                    buffer: s.buffer,
                    snapshot: s.tensor,
                    dst: u64::from(slot) * bytes,
                    src: (u64::from(slot) * u64::from(self.depth) + u64::from(committed - 1))
                        * bytes,
                    bytes,
                })
            })
            .collect()
    }
}

/// A tensor as the validator sees it: byte extent and whether the packet initializes it.
#[derive(Clone, Copy, Debug)]
pub struct TensorFacts {
    pub bytes: u64,
    pub initialized: bool,
}

impl Manifest {
    /// `tensor(handle)` describes the packet's tensor table (`None` = out of range); `slots` is
    /// the runtime's physical slot capacity.
    pub fn validate(
        &self,
        slots: u32,
        tensor: impl Fn(u16) -> Option<TensorFacts>,
    ) -> Result<(), String> {
        if self.version != VERSION {
            return Err(format!(
                "recurrent state version {} unsupported",
                self.version
            ));
        }
        if self.slots == 0 || self.slots != slots {
            return Err(format!(
                "recurrent state sized for {} slots, the runtime has {slots}",
                self.slots
            ));
        }
        let slot_major = |b: &Buffer, what: &str| -> Result<(), String> {
            let t = tensor(b.tensor)
                .ok_or_else(|| format!("{what}: tensor {} out of range", b.tensor))?;
            let want = u64::from(self.slots)
                .checked_mul(u64::from(b.slot_bytes))
                .filter(|_| b.slot_bytes > 0)
                .ok_or_else(|| format!("{what}: slot bytes {} invalid", b.slot_bytes))?;
            if t.bytes != want || t.initialized {
                return Err(format!(
                    "{what}: tensor {} is {} bytes, initialized={}; expected {} x {} uninitialized",
                    b.tensor, t.bytes, t.initialized, self.slots, b.slot_bytes
                ));
            }
            Ok(())
        };
        for (i, b) in self.buffers.iter().enumerate() {
            if self.buffers[..i].iter().any(|o| o.tensor == b.tensor) || b.tensor == self.active {
                return Err(format!("recurrent state tensor {} bound twice", b.tensor));
            }
            slot_major(b, "recurrent state buffer")?;
        }
        let active = tensor(self.active)
            .ok_or_else(|| format!("recurrent state active tensor {} out of range", self.active))?;
        let words = u64::from(self.slots) + u64::from(self.step_seed);
        if active.bytes != words * 4 || active.initialized {
            return Err(format!(
                "recurrent state active tensor {} must be an uninitialized i32[{words}]",
                self.active
            ));
        }
        if let Some(init) = &self.prefill_initstate {
            if init.tensor == self.active || self.buffers.iter().any(|b| b.tensor == init.tensor) {
                return Err(format!(
                    "recurrent state tensor {} bound twice",
                    init.tensor
                ));
            }
            slot_major(init, "recurrent state prefill initstate")?;
        }
        if let Some(v) = &self.verify {
            let kept = v
                .replay
                .as_ref()
                .map_or(v.snapshots.len(), |r| r.records.len());
            if v.depth == 0
                || kept != self.buffers.len()
                || (v.replay.is_some() && !v.snapshots.is_empty())
            {
                return Err(format!(
                    "recurrent verify: depth {} with {} snapshots, {kept} kept states for {} buffers",
                    v.depth,
                    v.snapshots.len(),
                    self.buffers.len()
                ));
            }
            let mut bound: Vec<u16> = self.buffers.iter().map(|b| b.tensor).collect();
            bound.push(self.active);
            bound.extend(self.prefill_initstate.as_ref().map(|b| b.tensor));
            if let Some(r) = &v.replay {
                let words = u64::from(self.slots) + 2;
                let t = tensor(r.table)
                    .ok_or_else(|| format!("recurrent replay: table {} out of range", r.table))?;
                if bound.contains(&r.table) || t.bytes != words * 4 || t.initialized {
                    return Err(format!(
                        "recurrent replay: table {} must be an unbound uninitialized i32[{words}]",
                        r.table
                    ));
                }
                bound.push(r.table);
                for rec in &r.records {
                    if !self.buffers.iter().any(|b| b.tensor == rec.buffer)
                        || r.records.iter().filter(|o| o.buffer == rec.buffer).count() != 1
                    {
                        return Err(format!(
                            "recurrent replay: record of {} invalid",
                            rec.buffer
                        ));
                    }
                    if bound.contains(&rec.tensor) {
                        return Err(format!(
                            "recurrent state tensor {} bound twice",
                            rec.tensor
                        ));
                    }
                    bound.push(rec.tensor);
                    let t = tensor(rec.tensor).ok_or_else(|| {
                        format!("recurrent replay: tensor {} out of range", rec.tensor)
                    })?;
                    let want = 2 * u64::from(self.slots) * u64::from(rec.row_bytes);
                    if rec.row_bytes == 0
                        || rec.row_bytes % 16 != 0
                        || t.bytes != want
                        || t.initialized
                    {
                        return Err(format!(
                            "recurrent replay: record {} is {} bytes, initialized={}; expected 2 x {} x {} (16-byte rows) uninitialized",
                            rec.tensor, t.bytes, t.initialized, self.slots, rec.row_bytes
                        ));
                    }
                }
            }
            for s in &v.snapshots {
                let b = self
                    .buffers
                    .iter()
                    .find(|b| b.tensor == s.buffer)
                    .ok_or_else(|| format!("recurrent verify: {} is not a buffer", s.buffer))?;
                if bound.contains(&s.tensor)
                    || v.snapshots.iter().filter(|o| o.buffer == s.buffer).count() != 1
                {
                    return Err(format!("recurrent state tensor {} bound twice", s.tensor));
                }
                bound.push(s.tensor);
                let t = tensor(s.tensor)
                    .ok_or_else(|| format!("recurrent verify: tensor {} out of range", s.tensor))?;
                let want = u64::from(self.slots) * u64::from(v.depth) * u64::from(b.slot_bytes);
                if t.bytes != want || t.initialized {
                    return Err(format!(
                        "recurrent verify: snapshot {} is {} bytes, initialized={}; expected {} x {} x {} uninitialized",
                        s.tensor, t.bytes, t.initialized, self.slots, v.depth, b.slot_bytes
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table<'a>(entries: &'a [(u64, bool)]) -> impl Fn(u16) -> Option<TensorFacts> + 'a {
        move |h| {
            entries
                .get(h as usize)
                .map(|&(bytes, initialized)| TensorFacts { bytes, initialized })
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            version: VERSION,
            slots: 4,
            buffers: vec![
                Buffer {
                    tensor: 0,
                    slot_bytes: 61440,
                },
                Buffer {
                    tensor: 1,
                    slot_bytes: 3145728,
                },
            ],
            active: 2,
            prefill_initstate: Some(Buffer {
                tensor: 3,
                slot_bytes: 3145728,
            }),
            prefill_updates_state: true,
            step_seed: false,
            verify: None,
        }
    }

    #[test]
    fn validates_slot_major_buffers_active_and_initstate() {
        let m = manifest();
        let ok = [
            (4 * 61440, false),
            (4 * 3145728, false),
            (16, false),
            (4 * 3145728, false),
        ];
        assert!(m.validate(4, table(&ok)).is_ok());
        assert!(m
            .validate(8, table(&ok))
            .unwrap_err()
            .contains("sized for 4 slots"));
        let short = [
            (3 * 61440, false),
            (4 * 3145728, false),
            (16, false),
            (4 * 3145728, false),
        ];
        assert!(m.validate(4, table(&short)).unwrap_err().contains("buffer"));
        let init = [
            (4 * 61440, true),
            (4 * 3145728, false),
            (16, false),
            (4 * 3145728, false),
        ];
        assert!(m
            .validate(4, table(&init))
            .unwrap_err()
            .contains("initialized=true"));
        let active = [
            (4 * 61440, false),
            (4 * 3145728, false),
            (8, false),
            (4 * 3145728, false),
        ];
        assert!(m
            .validate(4, table(&active))
            .unwrap_err()
            .contains("active"));
        let mut twice = m.clone();
        twice.buffers[1].tensor = 0;
        assert!(twice
            .validate(4, table(&ok))
            .unwrap_err()
            .contains("bound twice"));
        let mut range = m.clone();
        range.active = 9;
        assert!(range
            .validate(4, table(&ok))
            .unwrap_err()
            .contains("out of range"));
        // Active-only: a packet that takes the mask without persistent buffers.
        let mut none = m.clone();
        none.buffers.clear();
        none.prefill_initstate = None;
        assert!(none.validate(4, table(&ok)).is_ok());
        let mut seeded = m.clone();
        seeded.step_seed = true;
        assert!(seeded
            .validate(4, table(&ok))
            .unwrap_err()
            .contains("i32[5]"));
        let wide = [
            (4 * 61440, false),
            (4 * 3145728, false),
            (20, false),
            (4 * 3145728, false),
        ];
        assert!(seeded.validate(4, table(&wide)).is_ok());
        assert!(m.validate(4, table(&wide)).is_err());
        let json = serde_json::to_string(&seeded).unwrap();
        assert!(json.contains("\"step_seed\":true"));
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), seeded);
        assert!(!serde_json::to_string(&m).unwrap().contains("step_seed"));
        let mut v = m.clone();
        v.version = 7;
        assert!(v.validate(4, table(&ok)).unwrap_err().contains("version"));
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        assert!(serde_json::from_str::<Manifest>(&json.replace("\"slots\"", "\"slot\"")).is_err());
    }

    fn verify() -> (Manifest, Vec<(u64, bool)>) {
        let mut m = manifest();
        m.verify = Some(VerifyRollback {
            depth: 3,
            snapshots: vec![
                Snapshot {
                    buffer: 0,
                    tensor: 4,
                },
                Snapshot {
                    buffer: 1,
                    tensor: 5,
                },
            ],
            replay: None,
        });
        let t = vec![
            (4 * 61440, false),
            (4 * 3145728, false),
            (16, false),
            (4 * 3145728, false),
            (4 * 3 * 61440, false),
            (4 * 3 * 3145728, false),
        ];
        (m, t)
    }

    #[test]
    fn verify_snapshots_are_slot_major_per_depth_and_cover_every_buffer() {
        let (m, t) = verify();
        m.validate(4, table(&t)).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        let mut short = t.clone();
        short[5].0 = 4 * 2 * 3145728;
        assert!(m
            .validate(4, table(&short))
            .unwrap_err()
            .contains("snapshot"));
        let mut init = t.clone();
        init[4].1 = true;
        assert!(m.validate(4, table(&init)).is_err());
        let mut missing = m.clone();
        missing.verify.as_mut().unwrap().snapshots.pop();
        assert!(missing.validate(4, table(&t)).is_err());
        let mut alias = m.clone();
        alias.verify.as_mut().unwrap().snapshots[1].tensor = 3;
        assert!(alias
            .validate(4, table(&t))
            .unwrap_err()
            .contains("bound twice"));
        let mut twice = m.clone();
        twice.verify.as_mut().unwrap().snapshots[1].buffer = 0;
        assert!(twice.validate(4, table(&t)).is_err());
        let mut zero = m.clone();
        zero.verify.as_mut().unwrap().depth = 0;
        assert!(zero.validate(4, table(&t)).is_err());
        // A manifest without the field keeps its old JSON.
        assert!(!serde_json::to_string(&manifest())
            .unwrap()
            .contains("verify"));
    }

    #[test]
    fn verify_commit_copies_the_last_committed_row_unless_the_span_is_kept() {
        let (m, _) = verify();
        let v = m.verify.as_ref().unwrap();
        assert!(v.commit(&m.buffers, 2, 4, 4).unwrap().is_empty());
        let copies = v.commit(&m.buffers, 2, 4, 2).unwrap();
        assert_eq!(copies.len(), 2);
        assert_eq!(
            copies[1],
            CommitCopy {
                buffer: 1,
                snapshot: 5,
                dst: 2 * 3145728,
                src: (2 * 3 + 1) * 3145728,
                bytes: 3145728,
            }
        );
        assert_eq!(copies[0].src, (2 * 3) * 61440 + 61440);
        assert_eq!(v.commit(&m.buffers, 0, 2, 1).unwrap()[0].src, 0);
        for (rows, committed) in [(5, 1), (0, 0), (3, 0), (3, 4)] {
            assert!(v.commit(&m.buffers, 0, rows, committed).is_err());
        }
    }

    #[test]
    fn verify_replay_records_two_parities_per_slot_and_copies_nothing() {
        let (mut m, mut t) = verify();
        let v = m.verify.as_mut().unwrap();
        v.snapshots.clear();
        v.replay = Some(Replay {
            table: 4,
            records: vec![
                Record {
                    buffer: 0,
                    tensor: 5,
                    row_bytes: 20480,
                },
                Record {
                    buffer: 1,
                    tensor: 6,
                    row_bytes: 18944,
                },
            ],
        });
        t.truncate(4);
        t.extend([
            (6 * 4, false),
            (2 * 4 * 20480, false),
            (2 * 4 * 18944, false),
        ]);
        m.validate(4, table(&t)).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("replay"));
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        let v = m.verify.as_ref().unwrap();
        assert!(v.commit(&m.buffers, 3, 4, 2).unwrap().is_empty());
        assert!(v.commit(&m.buffers, 3, 4, 0).is_err());
        let mut short = t.clone();
        short[6].0 = 4 * 18944;
        assert!(m.validate(4, table(&short)).unwrap_err().contains("record"));
        let mut words = t.clone();
        words[4].0 = 16;
        assert!(m.validate(4, table(&words)).unwrap_err().contains("table"));
        let mut snap = m.clone();
        snap.verify.as_mut().unwrap().snapshots.push(Snapshot {
            buffer: 0,
            tensor: 5,
        });
        assert!(snap.validate(4, table(&t)).is_err());
        let mut missing = m.clone();
        missing
            .verify
            .as_mut()
            .unwrap()
            .replay
            .as_mut()
            .unwrap()
            .records
            .pop();
        assert!(missing.validate(4, table(&t)).is_err());
        let mut alias = m.clone();
        alias.verify.as_mut().unwrap().replay.as_mut().unwrap().records[1].tensor = 4;
        assert!(alias
            .validate(4, table(&t))
            .unwrap_err()
            .contains("bound twice"));
        let mut odd = m.clone();
        odd.verify.as_mut().unwrap().replay.as_mut().unwrap().records[1].row_bytes = 18940;
        let mut odd_t = t.clone();
        odd_t[6].0 = 2 * 4 * 18940;
        assert!(odd.validate(4, table(&odd_t)).is_err());
        assert_eq!(
            Replay::entry(3, 5, 1, true).unwrap(),
            3 | 5 << 8 | 1 << 16 | 1 << 17
        );
        assert!(Replay::entry(0, 0, 0, false).is_err());
        assert!(Replay::entry(1, 256, 0, false).is_err());
    }
}
