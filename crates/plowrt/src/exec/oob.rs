//! §OOB Out-of-band channel — bidirectional side-channel, distinct from the
//! data-plane packet queue.
//!
//! * **Host → executor (control):** `SWITCH_SCHEDULE`, `UPDATE_INDIRECTION`,
//!   `CANCEL`, `BARRIER`. Checked at iteration boundaries (cancel more often).
//! * **Executor → host (feedback):** `FAULT`, `CHECKPOINT` (counter/timestamp
//!   snapshot for §K tracing), `SPEC_VERDICT` (speculative accept length).
//!
//! Messages are small `#[repr(C)]` POD, same discipline as `packet`. The control
//! ring is cold (iteration boundary); the event ring is drained by a background
//! tokio task. NOTE: for the skeleton these are `parking_lot` rings; production
//! replaces them with the lock-free ring in `queue.rs` mapped over host-pinned
//! memory so `CHECKPOINT` emission from a worker never takes a lock.

use std::collections::VecDeque;

use parking_lot::Mutex;

/// OOB message kind (stable numeric ids, shared with the device ABI).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum OobKind {
    // host → executor
    SwitchSchedule = 0x01,
    UpdateIndirection = 0x02,
    Cancel = 0x03,
    Barrier = 0x04,
    // executor → host
    Fault = 0x81,
    Checkpoint = 0x82,
    SpecVerdict = 0x83,
}

/// One OOB record. `#[repr(C)]` POD so it maps byte-for-byte to the device side.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct OobMsg {
    pub kind: u16,
    pub _pad: u16,
    pub exec: u32,
    pub arg0: u64,
    pub arg1: u64,
}

impl OobMsg {
    #[inline]
    pub fn new(kind: OobKind, exec: u32, arg0: u64, arg1: u64) -> Self {
        OobMsg {
            kind: kind as u16,
            _pad: 0,
            exec,
            arg0,
            arg1,
        }
    }
}

/// One executor's bidirectional OOB endpoint.
pub struct OobChannel {
    control: Mutex<VecDeque<OobMsg>>, // host → executor
    events: Mutex<VecDeque<OobMsg>>,  // executor → host
}

impl Default for OobChannel {
    fn default() -> Self {
        OobChannel {
            control: Mutex::new(VecDeque::new()),
            events: Mutex::new(VecDeque::new()),
        }
    }
}

impl OobChannel {
    /// Host: post a control message (checked by the executor at a boundary).
    pub fn send_control(&self, msg: OobMsg) {
        self.control.lock().push_back(msg);
    }

    /// Executor: take the next pending control message, if any.
    pub fn recv_control(&self) -> Option<OobMsg> {
        self.control.lock().pop_front()
    }

    /// Executor: emit a feedback event (fault / checkpoint / verdict).
    pub fn emit_event(&self, msg: OobMsg) {
        self.events.lock().push_back(msg);
    }

    /// Host: drain all pending feedback events (background tokio task).
    pub fn drain_events(&self, out: &mut Vec<OobMsg>) {
        let mut q = self.events.lock();
        out.extend(q.drain(..));
    }

    /// Convenience: broadcast a `CANCEL` to unwedge a tenant (deadlock/watchdog).
    pub fn cancel(&self, exec: u32) {
        self.send_control(OobMsg::new(OobKind::Cancel, exec, 0, 0));
    }
}
