use super::*;

fn step(slots: &[usize]) -> PipeStep {
    PipeStep {
        buf: 0,
        rows: slots.iter().map(|&slot| PipeRow { slot, carry: true }).collect(),
        compact: false,
    }
}

/// E4B with prefix reuse off faulted at C64: slot 63 left the batch, the 64-row rung kept running
/// its row while the end-of-tick retire unmapped its live ring under the queued lookahead step.
#[test]
fn queued_step_defers_retiring_slots_it_does_not_feed() {
    let mut q = PipeQueue::new(64);
    assert!(!q.defer_retire(63, true), "nothing queued: retire now");

    q.steps.push_back(step(&(0..63).collect::<Vec<_>>()));
    assert!(!q.holds(63));
    assert!(q.defer_retire(63, false));
    assert!(q.defer_retire(5, true));
    assert!(q.take_retired().is_empty(), "held while a step is queued");

    q.steps.pop_front();
    assert_eq!(q.take_retired().as_slice(), &[(5, true), (63, false)]);
    assert!(q.take_retired().is_empty());
}

/// A failed step clears the queue but keeps its deferred retirements; the slot's direct retire
/// must drop them, or they fire later against the request seated there next.
#[test]
fn a_direct_retire_forgets_a_deferral_left_by_a_failed_step() {
    let mut q = PipeQueue::new(8);
    q.push(step(&[0]));
    assert!(q.defer_retire(7, true));
    q.pop();
    q.clear();
    assert!(!q.defer_retire(7, true), "nothing queued: retired directly");
    q.forget(7);
    q.push(step(&[7]));
    assert!(!q.retiring(7));
    q.pop();
    assert!(q.take_retired().is_empty());
}
