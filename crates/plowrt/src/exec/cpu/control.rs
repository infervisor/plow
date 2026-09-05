//! Host → worker control queue and worker → host feedback.
//!
//! A lock-free **broadcast** ring: the host is the single producer (`tail`),
//! every worker keeps its own read position and publishes it (`seen[w]`) so the
//! host never overwrites a slot a slow worker has not consumed. Modeled on
//! `runtime/common/control_queue_probe.h` (`host_tail_seq` / `dev_head_seq`),
//! with the same 64-byte command records. The ring is cold (one command per
//! run / reset / barrier); the per-packet path never touches it beyond one
//! relaxed load of `cancel_gen`.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crossbeam_utils::CachePadded;

pub const CMD_NOP: u32 = 0;
/// `a` = `*const LoadedProgram`, `b` = `*const CounterPool`, `c` = segment. `gen` = run id.
pub const CMD_RUN: u32 = 1;
/// Abandon run `gen` at the next packet boundary (host set `cancel_gen` first).
pub const CMD_CANCEL: u32 = 2;
/// Zero this worker's share of `[b, b + c)` bytes; `a` = slot (informational).
pub const CMD_RESET_SLOT: u32 = 3;
/// Acknowledge via `Feedback::barrier_ack`; `a` = barrier seq.
pub const CMD_BARRIER: u32 = 4;
/// Exit the thread. Sent only from `WorkerPool::drop`.
pub const CMD_STOP: u32 = 5;

/// One 64-byte control record (`#[repr(C)]`: mirrors the device-side shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Cmd {
    pub kind: u32,
    pub gen: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    _pad: [u64; 4],
}

const _: () = assert!(std::mem::size_of::<Cmd>() == 64);

impl Cmd {
    #[inline]
    pub fn new(kind: u32, gen: u32, a: u64, b: u64, c: u64) -> Cmd {
        Cmd {
            kind,
            gen,
            a,
            b,
            c,
            _pad: [0; 4],
        }
    }
    pub fn run(gen: u32, prog: u64, counters: u64, seg: u32) -> Cmd {
        Cmd::new(CMD_RUN, gen, prog, counters, seg as u64)
    }
    pub fn cancel(gen: u32) -> Cmd {
        Cmd::new(CMD_CANCEL, gen, 0, 0, 0)
    }
    pub fn reset_slot(slot: u32, ptr: u64, len: u64) -> Cmd {
        Cmd::new(CMD_RESET_SLOT, 0, slot as u64, ptr, len)
    }
    pub fn barrier(seq: u32) -> Cmd {
        Cmd::new(CMD_BARRIER, 0, seq as u64, 0, 0)
    }
    pub fn stop() -> Cmd {
        Cmd::new(CMD_STOP, 0, 0, 0, 0)
    }
}

pub const RING_CAPACITY: usize = 256;

pub struct ControlRing {
    slots: Box<[UnsafeCell<Cmd>]>,
    mask: u64,
    /// Next sequence the host will write; a slot is valid once `tail > seq`.
    tail: CachePadded<AtomicU64>,
    /// Per-worker consumed sequence (`seen[w]` = next seq worker w will read).
    seen: Box<[CachePadded<AtomicU64>]>,
}

// SAFETY: a slot is written by the single producer strictly before the Release
// store of `tail` that makes it visible, and is not rewritten until every
// worker's Release-published `seen` has moved past it. Readers copy the record
// out after an Acquire load of `tail`.
unsafe impl Send for ControlRing {}
unsafe impl Sync for ControlRing {}

impl ControlRing {
    pub fn new(workers: usize) -> ControlRing {
        let mut slots = Vec::with_capacity(RING_CAPACITY);
        slots.resize_with(RING_CAPACITY, || UnsafeCell::new(Cmd::default()));
        let mut seen = Vec::with_capacity(workers);
        seen.resize_with(workers, || CachePadded::new(AtomicU64::new(0)));
        ControlRing {
            slots: slots.into_boxed_slice(),
            mask: RING_CAPACITY as u64 - 1,
            tail: CachePadded::new(AtomicU64::new(0)),
            seen: seen.into_boxed_slice(),
        }
    }

    #[inline]
    fn min_seen(&self) -> u64 {
        self.seen
            .iter()
            .map(|s| s.load(Ordering::Acquire))
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Host: publish one command to every worker. Spins if the ring is full,
    /// which needs 256 unconsumed commands — it does not happen in practice.
    pub fn push(&self, cmd: Cmd) -> u64 {
        let tail = self.tail.load(Ordering::Relaxed);
        while tail.wrapping_sub(self.min_seen()) >= RING_CAPACITY as u64 {
            std::thread::yield_now();
        }
        // SAFETY: slot `tail` is past every worker's `seen` (checked above).
        unsafe { *self.slots[(tail & self.mask) as usize].get() = cmd };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        tail
    }

    /// Worker: the command at `seen`, if the host has published it.
    #[inline]
    pub fn peek(&self, seen: u64) -> Option<Cmd> {
        if self.tail.load(Ordering::Acquire) == seen {
            return None;
        }
        // SAFETY: `tail > seen` ⇒ the slot was fully written before the Release.
        Some(unsafe { *self.slots[(seen & self.mask) as usize].get() })
    }

    /// Worker: publish that it has consumed everything below `seen`.
    #[inline]
    pub fn ack(&self, worker: usize, seen: u64) {
        self.seen[worker].store(seen, Ordering::Release);
    }

    pub fn tail(&self) -> u64 {
        self.tail.load(Ordering::Acquire)
    }
}

/// Worker → host signals. No locks: the host polls these.
pub struct Feedback {
    /// Workers that have drained (or abandoned) the current run.
    pub done: CachePadded<AtomicU32>,
    /// Workers that have acknowledged the latest `CMD_BARRIER`.
    pub barrier_ack: CachePadded<AtomicU32>,
    /// First fault of the current run, `0` = none: see [`pack_fault`].
    pub fault: CachePadded<AtomicU64>,
    /// Run generation to abandon (`0` = none). Read once per packet, Relaxed.
    pub cancel_gen: CachePadded<AtomicU32>,
}

impl Default for Feedback {
    fn default() -> Self {
        Feedback {
            done: CachePadded::new(AtomicU32::new(0)),
            barrier_ack: CachePadded::new(AtomicU32::new(0)),
            fault: CachePadded::new(AtomicU64::new(0)),
            cancel_gen: CachePadded::new(AtomicU32::new(0)),
        }
    }
}

/// `op:16 | inst:32 | worker:16` — `+1` on op so a fault on op 0 is never `0`.
#[inline]
pub fn pack_fault(op: u16, inst: u32, worker: u16) -> u64 {
    ((op as u64 + 1) << 48) | ((inst as u64) << 16) | worker as u64
}

pub fn unpack_fault(v: u64) -> Option<(u16, u32, u16)> {
    if v == 0 {
        return None;
    }
    Some((((v >> 48) - 1) as u16, (v >> 16) as u32, v as u16))
}

impl Feedback {
    /// Record the first fault of a run; later ones are dropped (the first is
    /// the root cause, the rest are consequences).
    #[inline]
    pub fn fault(&self, op: u16, inst: u32, worker: u16) {
        let _ = self.fault.compare_exchange(
            0,
            pack_fault(op, inst, worker),
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_ring_delivers_to_every_worker_in_order() {
        let r = ControlRing::new(3);
        r.push(Cmd::barrier(1));
        r.push(Cmd::barrier(2));
        for w in 0..3 {
            let mut seen = 0;
            let mut got = Vec::new();
            while let Some(c) = r.peek(seen) {
                seen += 1;
                r.ack(w, seen);
                got.push(c.a);
            }
            assert_eq!(got, vec![1, 2]);
        }
        assert_eq!(r.min_seen(), 2);
    }

    #[test]
    fn fault_packing_roundtrips_and_keeps_first() {
        assert_eq!(unpack_fault(pack_fault(0, 7, 3)), Some((0, 7, 3)));
        assert_eq!(unpack_fault(pack_fault(146, u32::MAX, 65535)), Some((146, u32::MAX, 65535)));
        let f = Feedback::default();
        f.fault(5, 1, 2);
        f.fault(6, 9, 9);
        assert_eq!(unpack_fault(f.fault.load(Ordering::Acquire)), Some((5, 1, 2)));
    }
}
