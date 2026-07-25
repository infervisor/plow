//! §D Per-executor packet queue — a bounded lock-free ring.
//!
//! The host enqueues packet handles (offsets into the L2-resident `.pkt` bytes);
//! the executor's interpreter consumes them strictly FIFO. Single-producer /
//! single-consumer is the common case (the host feeds one executor's stream);
//! the design's MPSC variant (cross-tenant injection) generalizes the tail to a
//! `fetch_add` reservation — noted where it changes.
//!
//! Performance: power-of-two capacity so index→slot is a mask not a modulo;
//! head/tail on separate cache lines to avoid producer/consumer false sharing;
//! `Acquire`/`Release` handoff, no locks, no allocation after construction.

use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_utils::CachePadded;

/// A packet handle: an offset into the compiled `.pkt` byte stream.
pub type PacketHandle = u64;

pub struct PacketQueue {
    slots: Box<[AtomicU64]>,
    mask: u64,
    /// Next slot the producer will write (published on release).
    tail: CachePadded<AtomicU64>,
    /// Next slot the consumer will read.
    head: CachePadded<AtomicU64>,
}

impl PacketQueue {
    /// `capacity` is rounded up to a power of two (min 2). Sized large enough to
    /// hold a full inference iteration without back-pressure.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(2);
        let mut slots = Vec::with_capacity(cap);
        slots.resize_with(cap, || AtomicU64::new(0));
        PacketQueue {
            slots: slots.into_boxed_slice(),
            mask: cap as u64 - 1,
            tail: CachePadded::new(AtomicU64::new(0)),
            head: CachePadded::new(AtomicU64::new(0)),
        }
    }

    #[inline]
    fn capacity(&self) -> u64 {
        self.mask + 1
    }

    /// Producer: publish one handle. Returns `false` if the ring is full (the
    /// caller applies back-pressure). MPSC extension: reserve the slot with a
    /// `fetch_add` on `tail` instead of load+store.
    #[inline]
    pub fn push(&self, handle: PacketHandle) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= self.capacity() {
            return false;
        }
        self.slots[(tail & self.mask) as usize].store(handle, Ordering::Relaxed);
        // Release publishes the slot write to the consumer's Acquire.
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Consumer: pop the next handle in FIFO order, or `None` if empty.
    #[inline]
    pub fn pop(&self) -> Option<PacketHandle> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let handle = self.slots[(head & self.mask) as usize].load(Ordering::Relaxed);
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(handle)
    }

    /// Approximate depth — for §K queue-depth telemetry (racy by design, cheap).
    #[inline]
    pub fn depth(&self) -> u64 {
        self.tail
            .load(Ordering::Relaxed)
            .wrapping_sub(self.head.load(Ordering::Relaxed))
    }
}
