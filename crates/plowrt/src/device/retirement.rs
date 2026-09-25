use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SLOT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket {
    slot: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Ready,
    Prepared,
    InFlight,
    Canceled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RetirementError {
    Busy,
    Stale,
    State,
    Exhausted,
    IncompleteSignal(i64),
}

/// Completion is supplied only by the owning backend's matching wait/stream.
/// This is an admission guard, not a hardware completion proof.
pub(crate) struct RetirementSlot {
    id: u64,
    generation: u64,
    state: State,
}

impl RetirementSlot {
    pub(crate) fn new() -> Self {
        Self {
            id: NEXT_SLOT
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("retirement identity exhausted"),
            generation: 0,
            state: State::Ready,
        }
    }

    pub(crate) fn begin(&mut self) -> Result<Ticket, RetirementError> {
        if self.state != State::Ready {
            return Err(RetirementError::Busy);
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(RetirementError::Exhausted)?;
        self.state = State::Prepared;
        Ok(Ticket {
            slot: self.id,
            generation: self.generation,
        })
    }

    fn check(&self, ticket: Ticket) -> Result<(), RetirementError> {
        if ticket.slot != self.id || ticket.generation != self.generation {
            return Err(RetirementError::Stale);
        }
        Ok(())
    }

    pub(crate) fn submitted(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if self.state != State::Prepared {
            return Err(RetirementError::State);
        }
        self.state = State::InFlight;
        Ok(())
    }

    pub(crate) fn not_submitted(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if self.state != State::Prepared {
            return Err(RetirementError::State);
        }
        self.state = State::Ready;
        Ok(())
    }

    pub(crate) fn failed(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if self.state == State::Ready {
            return Err(RetirementError::State);
        }
        self.state = State::Failed;
        Ok(())
    }

    pub(crate) fn cancel(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        self.state = match self.state {
            State::Prepared => State::Ready,
            State::InFlight => State::Canceled,
            _ => return Err(RetirementError::State),
        };
        Ok(())
    }

    pub(crate) fn complete(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if !matches!(self.state, State::InFlight | State::Canceled) {
            return Err(RetirementError::State);
        }
        self.state = State::Ready;
        Ok(())
    }

    pub(crate) fn complete_signal(
        &mut self,
        ticket: Ticket,
        value: i64,
    ) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if value != 0 {
            self.failed(ticket)?;
            return Err(RetirementError::IncompleteSignal(value));
        }
        self.complete(ticket)
    }

    pub(crate) fn quiesced(&mut self, ticket: Ticket) -> Result<(), RetirementError> {
        self.check(ticket)?;
        if !matches!(
            self.state,
            State::InFlight | State::Canceled | State::Failed
        ) {
            return Err(RetirementError::State);
        }
        self.state = State::Ready;
        Ok(())
    }

    pub(crate) fn event_ticket(&self) -> Result<Option<Ticket>, RetirementError> {
        match self.state {
            State::Ready => Ok(None),
            State::InFlight | State::Canceled => Ok(self.pending()),
            _ => Err(RetirementError::State),
        }
    }

    pub(crate) fn pending(&self) -> Option<Ticket> {
        (self.state != State::Ready).then_some(Ticket {
            slot: self.id,
            generation: self.generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retirement_rejects_stale_and_foreign_completions() {
        let mut slot = RetirementSlot::new();
        let first = slot.begin().unwrap();
        slot.submitted(first).unwrap();
        assert_eq!(slot.begin(), Err(RetirementError::Busy));
        slot.complete(first).unwrap();
        let second = slot.begin().unwrap();
        slot.submitted(second).unwrap();
        assert_eq!(slot.complete(first), Err(RetirementError::Stale));
        let mut other = RetirementSlot::new();
        let foreign = other.begin().unwrap();
        assert_eq!(slot.complete(foreign), Err(RetirementError::Stale));
        assert_eq!(slot.begin(), Err(RetirementError::Busy));
        slot.complete(second).unwrap();
        assert_eq!(slot.complete(second), Err(RetirementError::State));
    }

    #[test]
    fn retirement_cancellation_is_not_completion() {
        let mut slot = RetirementSlot::new();
        let prepared = slot.begin().unwrap();
        slot.cancel(prepared).unwrap();
        let sent = slot.begin().unwrap();
        slot.submitted(sent).unwrap();
        slot.cancel(sent).unwrap();
        assert_eq!(slot.begin(), Err(RetirementError::Busy));
        assert_eq!(slot.not_submitted(sent), Err(RetirementError::State));
        slot.complete(sent).unwrap();
        assert!(slot.begin().is_ok());
    }

    #[test]
    fn retirement_uncertain_failure_needs_real_quiescence() {
        let mut slot = RetirementSlot::new();
        let rejected = slot.begin().unwrap();
        slot.not_submitted(rejected).unwrap();
        let uncertain = slot.begin().unwrap();
        slot.failed(uncertain).unwrap();
        assert_eq!(slot.begin(), Err(RetirementError::Busy));
        assert_eq!(slot.event_ticket(), Err(RetirementError::State));
        assert_eq!(slot.complete(uncertain), Err(RetirementError::State));
        assert_eq!(slot.quiesced(rejected), Err(RetirementError::Stale));
        slot.quiesced(uncertain).unwrap();
        let unrecorded = slot.begin().unwrap();
        slot.submitted(unrecorded).unwrap();
        slot.failed(unrecorded).unwrap();
        assert_eq!(slot.event_ticket(), Err(RetirementError::State));
        slot.quiesced(unrecorded).unwrap();
        assert!(slot.pending().is_none());
    }

    #[test]
    fn retirement_signal_requires_exact_zero_and_matching_generation() {
        for value in [-1, 1, 2] {
            let mut slot = RetirementSlot::new();
            let ticket = slot.begin().unwrap();
            slot.submitted(ticket).unwrap();
            assert_eq!(
                slot.complete_signal(ticket, value),
                Err(RetirementError::IncompleteSignal(value))
            );
            assert_eq!(slot.begin(), Err(RetirementError::Busy));
            assert_eq!(slot.complete_signal(ticket, 0), Err(RetirementError::State));
        }
        let mut slot = RetirementSlot::new();
        let old = slot.begin().unwrap();
        slot.submitted(old).unwrap();
        slot.complete_signal(old, 0).unwrap();
        let new = slot.begin().unwrap();
        slot.submitted(new).unwrap();
        assert_eq!(slot.complete_signal(old, 0), Err(RetirementError::Stale));
        slot.complete_signal(new, 0).unwrap();
    }

    #[test]
    #[ignore = "optimized CPU guard microbenchmark; excludes driver calls"]
    fn retirement_guard_cost() {
        use std::{hint::black_box, time::Instant};
        let mut samples = Vec::new();
        for _ in 0..9 {
            let mut slot = RetirementSlot::new();
            let start = Instant::now();
            for _ in 0..1_000_000 {
                let t = black_box(&mut slot).begin().unwrap();
                black_box(&mut slot).submitted(black_box(t)).unwrap();
                let t = black_box(&slot).event_ticket().unwrap().unwrap();
                black_box(&mut slot).complete(black_box(t)).unwrap();
            }
            samples.push(start.elapsed().as_nanos() as f64 / 1_000_000.0);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "retirement guard full submit/wait cycle median ns: {:.2}",
            samples[4]
        );
    }
}
