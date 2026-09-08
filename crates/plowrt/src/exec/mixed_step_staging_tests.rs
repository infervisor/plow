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
        let metadata = staging.pending_device_metadata().unwrap();
        assert_eq!(metadata.decode_slots, [2]);
        assert_eq!(metadata.prefill_spans[0].row0, 1);
        assert_eq!(metadata.parked, [0, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!((metadata.rows, metadata.decode_rows), (8, 1));
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

#[test]
fn completion_scatter_is_failure_atomic_and_reuses_storage() {
    let mut staging = MixedStepStaging::with_capacity(8, 2, 4);
    let mut frontiers = [0, 4, 8, 12];
    let d = [decode(2, 0, 7), decode(3, 1, 8)];
    let p = [prefill(1, 3, 4, &[10, 11])];
    let before = storage(&staging);
    staging.stage(&d, &p, &frontiers, 8, 64, 5).unwrap();
    let mut output = [u32::MAX; 2];

    assert_eq!(
        staging.finish_after_device_success(&mut frontiers, &[42], &mut output),
        Err(StageError::OutputRows {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(frontiers, [0, 4, 8, 12]);
    assert_eq!(output, [u32::MAX; 2]);
    assert!(staging.pending_plan().is_some());

    frontiers[1] = 5;
    assert!(matches!(
        staging.finish_after_device_success(&mut frontiers, &[42, 43], &mut output),
        Err(StageError::FrontierChanged { slot: 1, .. })
    ));
    assert_eq!(frontiers, [0, 5, 8, 12]);
    assert_eq!(output, [u32::MAX; 2]);
    assert!(staging.pending_plan().is_some());

    frontiers[1] = 4;
    staging
        .finish_after_device_success(&mut frontiers, &[42, 43], &mut output)
        .unwrap();
    assert_eq!(frontiers, [0, 6, 9, 13]);
    assert_eq!(output, [42, 43]);
    assert_eq!(storage(&staging), before);
}

// ================================================================================================
// Unified token batch staging
// ================================================================================================

mod token {
    use super::super::{
        fill_token_batch_words, StageError, TokenBatchLayout, TokenBatchStaging, SPAN_WORDS,
    };
    use plow_asset::token_batch::{Phase, Request, Selection};

    fn request(id: u32, slot: u32, phase: Phase, tokens: &[u32], prompt_len: u32) -> Request<'_> {
        Request {
            id,
            slot,
            state_slot: slot,
            generation: 7,
            phase,
            tokens,
            prompt_len,
            selection: Selection::default(),
        }
    }

    /// Frontiers advance only after the WHOLE chain succeeds, and only for rows consumed as
    /// input. A sampled token advances nothing until it is fed back.
    #[test]
    fn commit_publishes_every_frontier_or_none() {
        let mut staging = TokenBatchStaging::with_capacity(64, 4);
        let mut frontiers = [100u32, 70, 0];
        let generations = [7u32, 7, 7];
        let decode = [5u32];
        let prompt: Vec<u32> = (0..50).collect();
        let requests = [
            request(1, 0, Phase::Decode, &decode, 100),
            request(2, 1, Phase::Prefill, &prompt, 120),
        ];
        let plan = staging
            .stage(&requests, &frontiers, &generations, 64, 4096, 0)
            .unwrap();
        assert_eq!((plan.real_rows, plan.sample_rows), (51, 2));
        assert_eq!(frontiers, [100, 70, 0], "staging must not move a frontier");

        staging
            .commit_after_device_success(&mut frontiers, &generations)
            .unwrap();
        assert_eq!(frontiers, [101, 120, 0]);
        assert_eq!(
            staging.commit_after_device_success(&mut frontiers, &generations),
            Err(StageError::NoPendingPlan),
        );
    }

    /// Failure before commit: the tail failed, so nothing is published and the frontiers are
    /// exactly as they were. Physical KV may already have changed — that is a fault-path
    /// problem for the affected slots, not something a rollback can pretend away.
    #[test]
    fn a_discarded_submission_publishes_nothing() {
        let mut staging = TokenBatchStaging::with_capacity(64, 4);
        let mut frontiers = [3u32];
        let generations = [7u32];
        let decode = [5u32];
        staging
            .stage(
                &[request(1, 0, Phase::Decode, &decode, 3)],
                &frontiers,
                &generations,
                8,
                64,
                0,
            )
            .unwrap();
        staging.discard();
        assert_eq!(frontiers, [3]);
        assert_eq!(
            staging.commit_after_device_success(&mut frontiers, &generations),
            Err(StageError::NoPendingPlan),
        );
        assert_eq!(frontiers, [3], "a refused commit changes nothing");
    }

    /// A slot recycled while the step was in flight is refused BY GENERATION, and no frontier
    /// moves — including the frontiers of the requests that were still valid. Checking every
    /// span before mutating any is the whole point.
    #[test]
    fn a_recycled_slot_is_refused_and_no_frontier_moves() {
        let mut staging = TokenBatchStaging::with_capacity(64, 4);
        let mut frontiers = [100u32, 70];
        let generations = [7u32, 7];
        let decode = [5u32];
        let prompt: Vec<u32> = (0..50).collect();
        staging
            .stage(
                &[
                    request(1, 0, Phase::Decode, &decode, 100),
                    request(2, 1, Phase::Prefill, &prompt, 120),
                ],
                &frontiers,
                &generations,
                64,
                4096,
                0,
            )
            .unwrap();

        let recycled = [7u32, 8];
        assert_eq!(
            staging.commit_after_device_success(&mut frontiers, &recycled),
            Err(StageError::GenerationChanged {
                request: 2,
                slot: 1,
                expected: 7,
                actual: 8,
            }),
        );
        assert_eq!(frontiers, [100, 70], "slot 0 must not commit either");

        // A frontier that moved underneath the plan is refused the same way.
        let moved = [101u32, 70];
        let mut moved_mut = moved;
        assert_eq!(
            staging.commit_after_device_success(&mut moved_mut, &generations),
            Err(StageError::FrontierChanged {
                slot: 0,
                expected: 100,
                actual: 101,
            }),
        );
        assert_eq!(moved_mut, moved);
    }

    /// Delivery is by LOGICAL REQUEST, in sample order — never by row number, because packed
    /// rows, physical slots and compact output rows are three different numberings.
    #[test]
    fn delivery_maps_compact_ids_to_their_logical_requests() {
        let mut staging = TokenBatchStaging::with_capacity(64, 4);
        let frontiers = [100u32, 70, 0];
        let generations = [7u32, 7, 7];
        let decode = [5u32];
        let prompt: Vec<u32> = (0..50).collect();
        let chunk: Vec<u32> = (0..8).collect();
        staging
            .stage(
                &[
                    request(41, 0, Phase::Decode, &decode, 100),
                    request(42, 1, Phase::Prefill, &prompt, 120),
                    // An intermediate chunk: contributes rows to M and NOTHING to the delivery.
                    request(43, 2, Phase::Prefill, &chunk, 4096),
                ],
                &frontiers,
                &generations,
                64,
                4096,
                0,
            )
            .unwrap();

        let mut out = Vec::new();
        staging.deliver(&[900, 901], &mut out).unwrap();
        assert_eq!(out, vec![(41, 900), (42, 901)]);
        assert_eq!(
            staging.deliver(&[900], &mut out),
            Err(StageError::SampleRows {
                expected: 2,
                actual: 1
            }),
        );
    }

    /// The one shared filler writes the arrays at their FULL compiled extent, padding included:
    /// a device that reads a row past M must find `active == 0` there, not what the previous
    /// step left in the slab.
    #[test]
    fn the_filler_writes_the_whole_compiled_extent() {
        let mut staging = TokenBatchStaging::with_capacity(64, 4);
        let frontiers = [7u32, 0];
        let generations = [7u32, 7];
        let decode = [5u32];
        let chunk: Vec<u32> = (0..3).collect();

        let layout = TokenBatchLayout::new(8, 4, 4).unwrap();
        let mut words = vec![0xDEAD_BEEFu32; layout.words()];

        let plan = staging
            .stage(
                &[
                    request(1, 0, Phase::Decode, &decode, 7),
                    request(2, 1, Phase::Prefill, &chunk, 3),
                ],
                &frontiers,
                &generations,
                8,
                64,
                9,
            )
            .unwrap();
        fill_token_batch_words(&layout, &mut words, plan).unwrap();

        assert_eq!(&words[layout.input_ids.clone()], &[5, 0, 1, 2, 0, 0, 0, 0]);
        assert_eq!(&words[layout.positions.clone()], &[7, 0, 1, 2, 0, 0, 0, 0]);
        assert_eq!(&words[layout.active.clone()], &[1, 1, 1, 1, 0, 0, 0, 0]);
        // Span words are the PrefillSpan layout, verbatim — the token batch reuses the struct
        // rather than defining a second one that would need its own ABI lock.
        let first = &words[layout.span_table.start..][..SPAN_WORDS];
        assert_eq!(first, &[0, 1, 0, 0, 7, 8, 0, 9]);
        let second = &words[layout.span_table.start + SPAN_WORDS..][..SPAN_WORDS];
        assert_eq!(
            second,
            &[1, 3, 1, 1, 0, 3, 1, 9],
            "kv_row0 == 0 sets RESET_STATE"
        );
        assert_eq!(&words[layout.sample_rows.clone()][..2], &[0, 3]);

        // A slab that is too small, or a plan whose capacity disagrees with the layout, is
        // refused rather than partially written.
        let narrow = TokenBatchLayout::new(4, 4, 4).unwrap();
        let mut small = vec![0u32; narrow.words()];
        assert!(fill_token_batch_words(&narrow, &mut small, plan).is_err());
    }
}
