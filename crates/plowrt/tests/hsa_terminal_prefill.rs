#![cfg(feature = "hsa")]

use plowrt::serve::engine::AmdServe;
use std::path::PathBuf;

#[test]
fn packed_terminal_prefill_matches_native_decode_and_preserves_slots() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: requires PLOW_GPU_TEST=1, PLOW_FUSION=1 and PLOW_GPU_ASSETS");
        return;
    }
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let mut engine = AmdServe::load(
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .expect("load engine");
    assert!(engine.batch() >= 4);
    assert!(engine.mixed_step_rows(1, 1).is_some());
    let mut cases = vec![
        (vec![0, 1, 2, 3], vec![128; 4]),
        (vec![3, 1], vec![2305, 2689]),
        (vec![2, 0], vec![8191, 8175]),
    ];
    if engine.batch() >= 8 {
        cases.push(((0..8).collect(), vec![128; 8]));
        cases.push((vec![7, 2, 5], vec![128, 256, 512]));
    }
    for (slots, lengths) in cases {
        let prompts: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| (0..len).map(|t| 100 + ((t + i * 7) % 100) as u32).collect())
            .collect();
        let mut whole_prefix = Vec::new();
        for (&slot, prompt) in slots.iter().zip(&prompts) {
            engine.prefill(slot, &prompt[..prompt.len() - 1]).unwrap();
            let first = engine.step(slot, *prompt.last().unwrap()).unwrap();
            whole_prefix.push(first);
            engine.release(slot);
        }
        prepare_prefixes(&mut engine, &slots, &prompts);
        let first: Vec<_> = slots
            .iter()
            .zip(&prompts)
            .map(|(&slot, prompt)| {
                engine.finish_prefill_batch(&[], &[(slot, prompt)]).unwrap()[0].1
            })
            .collect();
        let expected: Vec<_> = slots
            .iter()
            .zip(&prompts)
            .zip(&first)
            .map(|((&slot, prompt), &token)| {
                let next = (prompt.len() + 1 < 8192).then(|| engine.step(slot, token).unwrap());
                (token, next)
            })
            .collect();
        for &slot in &slots {
            engine.release(slot);
        }
        let packed = prepare_prefixes(&mut engine, &slots, &prompts);
        if lengths.iter().all(|&len| len == 128) {
            assert_eq!(packed, 1, "all cold prefixes share one dispatch");
        }
        let before: Vec<_> = slots
            .iter()
            .map(|&slot| engine.prefill_frontier(slot))
            .collect();
        let first = (slots[0], prompts[0].as_slice());
        assert!(engine.finish_prefill_batch(&[], &[first, first]).is_err());
        assert!(engine
            .finish_prefill_batch(&[(slots[0], 42)], &[first])
            .is_err());
        assert!(engine
            .finish_prefill_batch(&[], &[(slots[0], &[])])
            .is_err());
        assert_eq!(
            slots
                .iter()
                .map(|&slot| engine.prefill_frontier(slot))
                .collect::<Vec<_>>(),
            before
        );
        let members: Vec<_> = slots
            .iter()
            .zip(&prompts)
            .map(|(&s, p)| (s, p.as_slice()))
            .collect();
        let output = engine.finish_prefill_batch(&[], &members).unwrap();
        assert_eq!(output.len(), slots.len());
        for ((&(slot, token), &expected_slot), &(first, next)) in
            output.iter().zip(&slots).zip(&expected)
        {
            assert_eq!(slot, expected_slot);
            assert_eq!(token, first, "terminal token at slot {slot}");
            assert_eq!(engine.prefill_frontier(slot), None);
            if let Some(next) = next {
                assert_eq!(
                    engine.step(slot, token).unwrap(),
                    next,
                    "next token at slot {slot}"
                );
            }
        }
        for &slot in &slots {
            engine.release(slot);
        }
        eprintln!(
            "whole-prefix first-token matches: {}/{}",
            output
                .iter()
                .zip(&whole_prefix)
                .filter(|((_, token), reference)| token == *reference)
                .count(),
            output.len()
        );
        eprintln!("terminal prefill passed slots={slots:?} lengths={lengths:?} packed={packed}");
    }
}

fn prepare_prefixes(engine: &mut AmdServe, slots: &[usize], prompts: &[Vec<u32>]) -> usize {
    for (&slot, prompt) in slots.iter().zip(prompts) {
        engine
            .prepare_packed_prefill_slot(slot, prompt, 512)
            .unwrap();
        assert_eq!(engine.prefill_frontier(slot), Some(0));
    }
    let mut packed = 0;
    while slots
        .iter()
        .zip(prompts)
        .any(|(&slot, prompt)| !engine.terminal_prefill_ready(slot, prompt))
    {
        let members: Vec<_> = slots
            .iter()
            .zip(prompts)
            .filter(|(&slot, _)| engine.packable_prefill_span(slot, 512).is_some())
            .map(|(&slot, prompt)| (slot, prompt.as_slice()))
            .collect();
        if members.len() >= 2 {
            engine.advance_packed_prefill(&members).unwrap();
            packed += 1;
        } else {
            let (&slot, prompt) = slots
                .iter()
                .zip(prompts)
                .find(|(&slot, prompt)| !engine.terminal_prefill_ready(slot, prompt))
                .unwrap();
            assert_eq!(
                engine.prefill_chunked_at_most(slot, prompt, 512).unwrap(),
                None
            );
        }
    }
    packed
}
