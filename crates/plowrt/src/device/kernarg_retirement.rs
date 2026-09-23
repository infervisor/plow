use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_QUEUE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn ring_on_dispatch(chain_active: bool, immediate: impl FnOnce() -> bool) -> bool {
    !chain_active || immediate()
}

pub(crate) fn ring_on_commit(immediate: bool) -> bool {
    !immediate
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Admission {
    queue: u64,
    issued: u64,
    retired: u64,
    count: u64,
    capacity: u64,
    completed: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Prefix {
    queue: u64,
    base: u64,
    next: u64,
    end: u64,
}

impl Admission {
    pub(crate) fn count(self) -> u64 {
        self.count
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Pending,
    Capacity,
    Exhausted,
    Stale,
    Incomplete(i64),
}

// Single host producer, like the HSA_QUEUE_TYPE_SINGLE queue. Publication and
// completion refer to the entire reserved batch, never its partially built prefix.
pub(crate) struct KernargRetirement {
    queue: u64,
    issued: AtomicU64,
    published: AtomicU64,
    retired: AtomicU64,
}

impl KernargRetirement {
    pub(crate) fn new() -> Self {
        Self {
            queue: NEXT_QUEUE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("kernarg queue identity exhausted"),
            issued: AtomicU64::new(0),
            published: AtomicU64::new(0),
            retired: AtomicU64::new(0),
        }
    }

    pub(crate) fn preflight(
        &self,
        count: u64,
        capacity: u64,
        completion: impl FnOnce() -> i64,
    ) -> Result<Admission, Error> {
        let issued = self.completion_ticket()?;
        let retired = self.retired.load(Ordering::Acquire);
        let completed = !self.available(count, capacity)?;
        if completed {
            let signal = completion();
            if signal != 0 {
                return Err(Error::Incomplete(signal));
            }
        }
        let ticket = Admission {
            queue: self.queue,
            issued,
            retired,
            count,
            capacity,
            completed,
        };
        self.validate(ticket)?;
        Ok(ticket)
    }

    fn validate(&self, ticket: Admission) -> Result<(), Error> {
        if ticket.queue != self.queue
            || ticket.issued != self.completion_ticket()?
            || ticket.retired != self.retired.load(Ordering::Acquire)
        {
            return Err(Error::Stale);
        }
        Ok(())
    }

    pub(crate) fn reserve_admitted(&self, ticket: Admission) -> Result<u64, Error> {
        self.validate(ticket)?;
        if ticket.completed {
            self.complete(ticket.issued, 0)?;
        }
        self.reserve(ticket.count, ticket.capacity)
    }

    pub(crate) fn quiescent(&self) -> Result<(), Error> {
        if self.completion_ticket()? != self.retired.load(Ordering::Acquire) {
            return Err(Error::Pending);
        }
        Ok(())
    }

    pub(crate) fn prefix(
        &self,
        base: u64,
        next: u64,
        end: u64,
        immediate: bool,
    ) -> Result<Prefix, Error> {
        if !immediate
            || base > next
            || next > end
            || base == end
            || self.published.load(Ordering::Acquire) != base
            || self.issued.load(Ordering::Acquire) != end
        {
            return Err(Error::Pending);
        }
        Ok(Prefix {
            queue: self.queue,
            base,
            next,
            end,
        })
    }

    pub(crate) fn complete_prefix(
        &self,
        ticket: Prefix,
        next: u64,
        signal: i64,
    ) -> Result<(), Error> {
        if ticket.queue != self.queue || ticket.next != next {
            return Err(Error::Stale);
        }
        self.prefix(ticket.base, next, ticket.end, true)?;
        if signal != 0 {
            if signal < 0 {
                self.cancel();
            }
            return Err(Error::Incomplete(signal));
        }
        // A timed prefix can finish while the rest of this reservation is unissued.
        // It must never retire/reuse any part of the reserved kernarg batch.
        Ok(())
    }

    pub(crate) fn available(&self, count: u64, capacity: u64) -> Result<bool, Error> {
        let issued = self.issued.load(Ordering::Acquire);
        if issued != self.published.load(Ordering::Acquire) {
            return Err(Error::Pending);
        }
        if count == 0 || count > capacity {
            return Err(Error::Capacity);
        }
        // The HSA chain implementation reserves the final two u64 values.
        if issued
            .checked_add(count)
            .filter(|&end| end < u64::MAX - 1)
            .is_none()
        {
            return Err(Error::Exhausted);
        }
        Ok(issued - self.retired.load(Ordering::Acquire) <= capacity - count)
    }

    pub(crate) fn reserve(&self, count: u64, capacity: u64) -> Result<u64, Error> {
        if !self.available(count, capacity)? {
            return Err(Error::Capacity);
        }
        let base = self.issued.load(Ordering::Relaxed);
        self.issued.store(base + count, Ordering::Release);
        Ok(base)
    }

    pub(crate) fn publish(&self, end: u64) -> Result<(), Error> {
        if end != self.issued.load(Ordering::Acquire)
            || end <= self.published.load(Ordering::Acquire)
        {
            return Err(Error::Stale);
        }
        self.published.store(end, Ordering::Release);
        Ok(())
    }

    pub(crate) fn completion_ticket(&self) -> Result<u64, Error> {
        let ticket = self.published.load(Ordering::Acquire);
        if ticket != self.issued.load(Ordering::Acquire) {
            return Err(Error::Pending);
        }
        Ok(ticket)
    }

    pub(crate) fn cancel(&self) {
        // Reserved AQL positions cannot be rolled back. Never revive this batch.
        self.published.store(u64::MAX, Ordering::Release);
    }

    pub(crate) fn complete(&self, ticket: u64, signal: i64) -> Result<(), Error> {
        if ticket != self.completion_ticket()? {
            return Err(Error::Stale);
        }
        if signal != 0 {
            if signal < 0 {
                self.cancel();
            }
            return Err(Error::Incomplete(signal));
        }
        self.retired.store(ticket, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_prefix_never_retires_reservation_or_authorizes_rebinding() {
        let ring = KernargRetirement::new();
        ring.quiescent().unwrap();
        ring.reserve(4, 4).unwrap();
        assert_eq!(ring.quiescent(), Err(Error::Pending));
        assert!(ring.prefix(0, 0, 4, false).is_err());
        let empty = ring.prefix(0, 0, 4, true).unwrap();
        ring.complete_prefix(empty, 0, 0).unwrap();
        let prefix = ring.prefix(0, 2, 4, true).unwrap();
        assert_eq!(
            ring.complete_prefix(prefix, 2, 1),
            Err(Error::Incomplete(1))
        );
        ring.complete_prefix(prefix, 2, 0).unwrap();
        assert_eq!(ring.retired.load(Ordering::Relaxed), 0);
        assert_eq!(ring.reserve(1, 4), Err(Error::Pending));
        assert_eq!(ring.quiescent(), Err(Error::Pending));
        assert_eq!(ring.complete_prefix(prefix, 3, 0), Err(Error::Stale));
        let other = KernargRetirement::new();
        other.reserve(4, 4).unwrap();
        assert_eq!(other.complete_prefix(prefix, 2, 0), Err(Error::Stale));
        ring.publish(4).unwrap();
        assert_eq!(ring.complete_prefix(prefix, 2, 0), Err(Error::Pending));
        assert_eq!(ring.quiescent(), Err(Error::Pending));
        ring.complete(4, 0).unwrap();
        ring.quiescent().unwrap();
        ring.reserve(4, 4).unwrap();
        assert_eq!(ring.complete_prefix(prefix, 2, 0), Err(Error::Pending));
        let new_prefix = ring.prefix(4, 5, 8, true).unwrap();
        ring.cancel();
        assert_eq!(ring.complete_prefix(new_prefix, 5, 0), Err(Error::Pending));
        assert_eq!(ring.quiescent(), Err(Error::Pending));
    }

    #[test]
    fn negative_prefix_completion_poison_is_permanent() {
        let ring = KernargRetirement::new();
        ring.reserve(2, 4).unwrap();
        assert!(ring.prefix(0, 3, 2, true).is_err());
        assert!(ring.prefix(1, 1, 2, true).is_err());
        let prefix = ring.prefix(0, 1, 2, true).unwrap();
        assert_eq!(
            ring.complete_prefix(prefix, 1, -1),
            Err(Error::Incomplete(-1))
        );
        assert_eq!(ring.complete_prefix(prefix, 1, 0), Err(Error::Pending));
        assert_eq!(ring.publish(2), Err(Error::Stale));
    }

    #[test]
    fn wrap_requires_actual_completion_not_packet_consumption() {
        let ring = KernargRetirement::new();
        for base in 0..4 {
            assert_eq!(ring.reserve(1, 4), Ok(base));
            ring.publish(base + 1).unwrap();
        }
        assert_eq!(ring.reserve(1, 4), Err(Error::Capacity));
        let ticket = ring.completion_ticket().unwrap();
        assert_eq!(ring.complete(ticket, 1), Err(Error::Incomplete(1)));
        assert_eq!(ring.reserve(1, 4), Err(Error::Capacity));
        ring.complete(ticket, 0).unwrap();
        assert_eq!(ring.reserve(4, 4), Ok(4));
        ring.publish(8).unwrap();
        assert_eq!(ring.complete(ticket, 0), Err(Error::Stale));
        assert_eq!(ring.reserve(1, 4), Err(Error::Capacity));
        ring.complete(8, 0).unwrap();
        assert_eq!(ring.reserve(4, 4), Ok(8));
    }

    #[test]
    fn whole_chain_preflight_and_canceled_or_failed_chain_remain_pending() {
        let ring = KernargRetirement::new();
        ring.reserve(2, 4).unwrap();
        ring.publish(2).unwrap();
        assert_eq!(ring.reserve(3, 4), Err(Error::Capacity));
        assert_eq!(ring.reserve(2, 4), Ok(2));
        // Partial preparation, cancellation or failed publication is not completion.
        assert_eq!(ring.publish(3), Err(Error::Stale));
        assert_eq!(ring.completion_ticket(), Err(Error::Pending));
        assert_eq!(ring.complete(2, 0), Err(Error::Pending));
        assert_eq!(ring.reserve(1, 4), Err(Error::Pending));
        ring.publish(4).unwrap();
        ring.complete(4, 0).unwrap();
        assert_eq!(ring.reserve(4, 4), Ok(4));
    }

    #[test]
    fn stale_zero_before_new_reservation_cannot_retire_new_batch() {
        let ring = KernargRetirement::new();
        let stale = ring.completion_ticket().unwrap();
        ring.reserve(1, 4).unwrap();
        ring.publish(1).unwrap();
        assert_eq!(ring.complete(stale, 0), Err(Error::Stale));
        assert_eq!(ring.reserve(4, 4), Err(Error::Capacity));
        assert_eq!(ring.reserve(0, 4), Err(Error::Capacity));
        ring.complete(1, 0).unwrap();
        ring.issued.store(u64::MAX - 3, Ordering::Relaxed);
        ring.published.store(u64::MAX - 3, Ordering::Relaxed);
        ring.retired.store(u64::MAX - 3, Ordering::Relaxed);
        assert_eq!(ring.reserve(2, 4), Err(Error::Exhausted));
    }

    #[test]
    fn canceled_chain_cannot_publish_or_retire_even_after_zero() {
        let ring = KernargRetirement::new();
        ring.reserve(4, 4).unwrap();
        ring.cancel();
        assert_eq!(ring.publish(4), Err(Error::Stale));
        assert_eq!(ring.complete(4, 0), Err(Error::Pending));
        assert_eq!(ring.reserve(1, 4), Err(Error::Pending));

        let corrupt = KernargRetirement::new();
        corrupt.reserve(1, 4).unwrap();
        corrupt.publish(1).unwrap();
        assert_eq!(corrupt.complete(1, -1), Err(Error::Incomplete(-1)));
        assert_eq!(corrupt.complete(1, 0), Err(Error::Pending));
        assert_eq!(corrupt.reserve(1, 4), Err(Error::Pending));
    }

    #[test]
    #[ignore]
    fn admission_cost() {
        use std::{hint::black_box, time::Instant};
        let mut samples = Vec::new();
        for _ in 0..9 {
            let ring = KernargRetirement::new();
            let start = Instant::now();
            for _ in 0..1_000_000 {
                let base = match black_box(&ring).reserve(1, 4096) {
                    Ok(base) => base,
                    Err(Error::Capacity) => {
                        let ticket = black_box(&ring).completion_ticket().unwrap();
                        black_box(&ring).complete(ticket, black_box(0)).unwrap();
                        black_box(&ring).reserve(1, 4096).unwrap()
                    }
                    Err(error) => panic!("{error:?}"),
                };
                black_box(&ring).publish(black_box(base + 1)).unwrap();
            }
            samples.push(start.elapsed().as_nanos() as f64 / 1_000_000.0);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "kernarg admission median {:.2} ns/dispatch; samples {samples:?}",
            samples[4]
        );
    }
}
