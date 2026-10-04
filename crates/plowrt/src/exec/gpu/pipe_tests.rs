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
