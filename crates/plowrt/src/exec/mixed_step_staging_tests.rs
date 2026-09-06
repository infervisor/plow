use super::*;

fn decode(slot: u32, state_slot: u32, token: u32) -> DecodeRequest {
    DecodeRequest {
        slot,
        state_slot,
        token,
    }
}

fn prefill<'a>(slot: u32, state_slot: u32, start: u32, tokens: &'a [u32]) -> PrefillRequest<'a> {
    PrefillRequest {
        slot,
        state_slot,
        start,
        tokens,
        prompt_len: 64,
    }
}

fn storage(
    staging: &MixedStepStaging,
) -> (
    (*const (), *const (), *const (), *const (), *const ()),
    (usize, usize, usize, usize, usize),
) {
    let plan = &staging.plan;
    (
        (
            plan.rows.as_ptr().cast(),
            plan.decode_slots.as_ptr().cast(),
            plan.prefill_spans.as_ptr().cast(),
            plan.parked.as_ptr().cast(),
            plan.mapped_ends.as_ptr().cast(),
        ),
        (
            plan.rows.capacity(),
            plan.decode_slots.capacity(),
            plan.prefill_spans.capacity(),
            plan.parked.capacity(),
            plan.mapped_ends.capacity(),
        ),
    )
}

#[test]
fn reuses_all_storage_across_successful_steps() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let mut frontiers = [0, 4, 8, 12];
    let before = storage(&staging);

    for _ in 0..2 {
        let p_tokens = [10, 11];
        let d = [decode(2, 0, 7)];
        let p = [prefill(1, 3, frontiers[1], &p_tokens)];
        let plan = staging.stage(&d, &p, &frontiers, 8, 64, 5).unwrap();
        assert_eq!(plan.decode_rows, 1);
        assert_eq!(plan.decode_slots, [2]);
        assert_eq!(plan.real_rows, 3);
        assert_eq!(storage(&staging), before);
        staging.commit_after_device_success(&mut frontiers).unwrap();
        assert_eq!(storage(&staging), before);
    }

    assert_eq!(frontiers, [0, 8, 10, 12]);
}

#[test]
fn planning_failure_clears_state_without_losing_capacity() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let mut frontiers = [0, 4, 8, 12];
    let before = storage(&staging);
    let d = [decode(2, 0, 7)];

    staging.stage(&d, &[], &frontiers, 8, 64, 5).unwrap();
    staging.discard();
    let duplicate = [prefill(2, 1, 8, &[8])];
    assert!(matches!(
        staging.stage(&d, &duplicate, &frontiers, 8, 64, 5),
        Err(StageError::Plan(_))
    ));
    assert!(staging.pending_plan().is_none());
    assert!(staging.plan.rows.is_empty());
    assert_eq!(storage(&staging), before);
    assert_eq!(
        staging.commit_after_device_success(&mut frontiers),
        Err(StageError::NoPendingPlan)
    );
    assert_eq!(frontiers, [0, 4, 8, 12]);
}

#[test]
fn pending_submission_cannot_be_overwritten() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let frontiers = [0, 4, 8, 12];
    let d = [decode(2, 0, 7)];

    staging.stage(&d, &[], &frontiers, 8, 64, 5).unwrap();
    let before = staging.pending_plan().unwrap().rows.as_ptr();
    assert!(matches!(
        staging.stage(&[], &[prefill(1, 3, 4, &[10])], &frontiers, 8, 64, 5),
        Err(StageError::PendingPlan)
    ));
    assert_eq!(staging.pending_plan().unwrap().rows.as_ptr(), before);
    assert_eq!(staging.pending_plan().unwrap().rows[0].slot, 2);
}

#[test]
fn frontiers_change_only_after_explicit_success_commit() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let mut frontiers = [0, 4, 8, 12];
    let p_tokens = [10, 11];
    let d = [decode(2, 0, 7)];
    let p = [prefill(1, 3, 4, &p_tokens)];

    let plan = staging.stage(&d, &p, &frontiers, 8, 64, 5).unwrap();
    assert_eq!(frontiers, [0, 4, 8, 12]);
    assert!(plan
        .mapped_ends
        .iter()
        .any(|&(slot, end)| slot == 1 && end > 6));
    staging.discard();
    assert_eq!(frontiers, [0, 4, 8, 12]);

    staging.stage(&d, &p, &frontiers, 8, 64, 5).unwrap();
    staging.commit_after_device_success(&mut frontiers).unwrap();
    assert_eq!(frontiers, [0, 6, 9, 12]);
    assert!(staging.pending_plan().is_none());
}

#[test]
fn stale_frontier_rejects_the_whole_commit() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let mut frontiers = [0, 4, 8, 12];
    let p_tokens = [10, 11];
    let d = [decode(2, 0, 7)];
    let p = [prefill(1, 3, 4, &p_tokens)];

    staging.stage(&d, &p, &frontiers, 8, 64, 5).unwrap();
    frontiers[1] = 5;
    assert_eq!(
        staging.commit_after_device_success(&mut frontiers),
        Err(StageError::FrontierChanged {
            slot: 1,
            expected: 4,
            actual: 5,
        })
    );
    assert_eq!(frontiers, [0, 5, 8, 12]);
    assert!(staging.pending_plan().is_some());
}
