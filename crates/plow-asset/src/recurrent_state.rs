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
//! `prefill_initstate`, when present, is the gathered initial-state scratch of a packed external
//! prompt scan (`[slots][slot_bytes]`). `prefill_updates_state` says every prefill program
//! advances every buffer for every prompt token it consumes; without it a prompt is consumed by
//! the decode program. A packet whose tensor table has a `state.` class tensor or the active mask
//! input carries this section, and every `state.` class tensor is one of its buffers; a packet
//! without it, or with a `state.` class tensor it does not declare, is refused.
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "recurrent_state";
pub const VERSION: u32 = 1;

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
        if active.bytes != u64::from(self.slots) * 4 || active.initialized {
            return Err(format!(
                "recurrent state active tensor {} must be an uninitialized i32[{}]",
                self.active, self.slots
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
        let mut v = m.clone();
        v.version = 7;
        assert!(v.validate(4, table(&ok)).unwrap_err().contains("version"));
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        assert!(serde_json::from_str::<Manifest>(&json.replace("\"slots\"", "\"slot\"")).is_err());
    }
}
