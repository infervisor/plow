#![cfg(feature = "hsa")]

use plowrt::serve::engine::AmdServe;
use std::path::PathBuf;

#[test]
fn single_gpu_chunked_packed_and_multistep_match_isolated() {
    packed_parity([777, 901], 2, false, 128);
}

#[test]
fn packed_prefill_wraps_rings_at_unequal_positions() {
    packed_parity([2305, 2689], 16, true, 128);
}

#[test]
fn packed_prefill_full_rung_matches_isolated() {
    packed_parity([2305, 2689], 2, true, 512);
}

fn packed_parity(lengths: [usize; 2], rounds: usize, staggered: bool, chunk: u32) {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: requires PLOW_GPU_TEST=1 and PLOW_GPU_ASSETS");
        return;
    }
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).unwrap();
    assert_eq!(blob.tp.map_or(1, |tp| tp.n_gpu), 1);
    let mut engine = AmdServe::load(
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .expect("load single GPU engine");
    assert!(engine.batch() >= 4, "compile a decode ladder through B4");
    let prompts: Vec<Vec<u32>> = lengths
        .into_iter()
        .enumerate()
        .map(|(i, len)| (0..len).map(|t| 100 + ((t + i * 7) % 100) as u32).collect())
        .collect();
    let slots = [0, 3];
    const QUANTUM: usize = 4;
    let mut references = Vec::new();
    for (prompt, &slot) in prompts.iter().zip(&slots) {
        let first = engine.prefill(slot, prompt).expect("isolated prefill");
        let mut tokens = vec![first];
        for _ in 0..2 * QUANTUM {
            tokens.push(
                engine
                    .step(slot, *tokens.last().unwrap())
                    .expect("single decode"),
            );
        }
        references.push(tokens);
        engine.release(slot);
    }

    for (prompt, &slot) in prompts.iter().zip(&slots) {
        assert_eq!(
            engine
                .prefill_chunked_at_most(slot, prompt, chunk)
                .expect("first chunk"),
            None
        );
        assert_eq!(engine.prefill_frontier(slot), Some(chunk as usize));
    }
    if staggered {
        assert_eq!(
            engine
                .prefill_chunked_at_most(slots[1], &prompts[1], chunk)
                .unwrap(),
            None
        );
    }
    let before = slots.map(|slot| engine.prefill_frontier(slot));
    assert!(engine
        .advance_packed_prefill(&[(slots[0], &prompts[0]), (slots[0], &prompts[0])])
        .is_err());
    assert!(engine
        .advance_packed_prefill(&[(slots[0], &prompts[0]), (slots[1], &[])])
        .is_err());
    assert_eq!(slots.map(|slot| engine.prefill_frontier(slot)), before);
    for _ in 0..rounds {
        for &slot in &slots {
            let span = engine
                .packable_prefill_span(slot, 1024)
                .expect("packable cursor");
            assert!(engine.prefill_prog_t(span.program as usize).unwrap() >= 2 * chunk);
        }
        engine
            .advance_packed_prefill(&[
                (slots[0], prompts[0].as_slice()),
                (slots[1], prompts[1].as_slice()),
            ])
            .expect("packed middle chunks");
    }
    for (i, &slot) in slots.iter().enumerate() {
        assert_eq!(
            engine.prefill_frontier(slot),
            Some(before[i].unwrap() + rounds * chunk as usize)
        );
    }

    let mut feeds = Vec::new();
    for ((prompt, &slot), reference) in prompts.iter().zip(&slots).zip(&references) {
        let first = loop {
            if let Some(token) = engine
                .prefill_chunked_at_most(slot, prompt, chunk)
                .expect("remaining chunks")
            {
                break token;
            }
        };
        assert_eq!(first, reference[0], "packed prefill slot {slot}");
        feeds.push((slot, first));
    }
    assert_eq!(engine.multistep_quantum(&feeds, 1), None);
    assert_eq!(
        engine.multistep_quantum(&[feeds[0], feeds[0]], QUANTUM),
        None
    );
    assert_eq!(engine.multistep_quantum(&[(1, 100)], QUANTUM), None);
    assert_eq!(
        engine.multistep_quantum(&[(engine.batch(), 100)], QUANTUM),
        None
    );
    let mut captured = Vec::new();
    for round in 0..2 {
        assert_eq!(engine.multistep_quantum(&feeds, QUANTUM), Some(QUANTUM));
        assert_eq!(
            engine.multi_step(&feeds, QUANTUM, &mut captured).unwrap(),
            QUANTUM
        );
        for (i, &(slot, _)) in feeds.iter().enumerate() {
            assert_eq!(
                &captured[slot * QUANTUM..(slot + 1) * QUANTUM],
                &references[i][1 + round * QUANTUM..1 + (round + 1) * QUANTUM],
                "multistep round {round}, slot {slot}"
            );
        }
        for (slot, token) in &mut feeds {
            *token = captured[(*slot + 1) * QUANTUM - 1];
        }
    }
    for &slot in &slots {
        engine.release(slot);
    }
    assert_eq!(
        engine
            .prefill_chunked_at_most(slots[0], &prompts[0], chunk)
            .unwrap(),
        None
    );
    engine.release(slots[0]);
    assert_eq!(engine.prefill_frontier(slots[0]), None);
    assert_eq!(
        engine.prefill(slots[0], &prompts[0]).unwrap(),
        references[0][0]
    );
}
