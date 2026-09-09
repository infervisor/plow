#![cfg(feature = "hsa")]

use plow_asset::mixed_step::{DecodeRequest, PrefillRequest};
use plowrt::device::hsa::HsaBackend;
use plowrt::exec::amd::AmdEngine;
use std::{path::PathBuf, sync::Arc};

fn ordinary_assets() -> Option<(PathBuf, Vec<u32>, usize)> {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: requires PLOW_GPU_TEST=1 and ordinary GPU assets");
        return None;
    }
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).unwrap();
    let batch = blob.decode_progs().last().unwrap().t;
    let mut buckets: Vec<_> = blob
        .prefill_progs()
        .iter()
        .filter(|program| !program.packed_prefill_only && program.t > batch)
        .map(|program| program.t)
        .collect();
    buckets.sort_unstable();
    buckets.dedup();
    assert!(!buckets.is_empty(), "ordinary prefill bucket inventory");
    let ring = blob
        .decode_progs()
        .iter()
        .flat_map(|program| &program.insts)
        .find(|inst| inst.op == packet::dev::DevOp::FlashDecode as u16 && inst.i[4] != 0)
        .expect("windowed ordinary FlashDecode")
        .i[3] as usize;
    assert!(
        ring >= 64 && ring.is_power_of_two(),
        "sliding KV ring capacity"
    );
    Some((assets, buckets, ring))
}

#[test]
fn fusion_disabled_preserves_ordinary_inference() {
    let Some((assets, buckets, _)) = ordinary_assets() else {
        return;
    };
    if std::env::var_os("PLOW_TEST_FUSION_DISABLED_CHILD").is_none() {
        // Runtime configuration is process-wide; the child inherits the parent's GPU lease.
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "fusion_disabled_preserves_ordinary_inference",
                "--exact",
                "--nocapture",
            ])
            .env("PLOW_FUSION", "0")
            .env("PLOW_TEST_FUSION_DISABLED_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "fusion-disabled child:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let be = Arc::new(HsaBackend::new(0).expect("leased HSA device"));
    let mut engine = AmdEngine::load(
        be,
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .unwrap();
    for rows in &buckets {
        assert_eq!(engine.mixed_step_rows(1, *rows as usize - 1), None);
    }
    let prompt = prompt(97, 0);
    let first = engine.prefill_slot(0, &prompt).unwrap();
    let expected = isolated_decode(&mut engine, first, prompt.len() as u32);
    let expected_logits = logits(&engine);
    assert_eq!(engine.prefill_slot(0, &prompt).unwrap(), first);
    let decode = [DecodeRequest {
        slot: 0,
        state_slot: 0,
        token: first,
    }];
    let prefill = [PrefillRequest {
        slot: 1,
        state_slot: 1,
        start: 0,
        tokens: &[100],
        prompt_len: 2,
    }];
    let mut frontiers = vec![0; engine.batch()];
    frontiers[0] = prompt.len() as u32;
    let before = frontiers.clone();
    let mut output = [u32::MAX];
    let launches = engine.seg_launches;
    assert!(engine
        .mixed_step(buckets[0], &decode, &prefill, &mut frontiers, &mut output)
        .is_err());
    assert_eq!(
        engine.seg_launches, launches,
        "disabled fusion rejects before launch"
    );
    assert_eq!(frontiers, before);
    assert_eq!(output, [u32::MAX]);
    assert_eq!(
        isolated_decode(&mut engine, first, prompt.len() as u32),
        expected
    );
    assert!(
        logits(&engine) == expected_logits,
        "ordinary inference survives disabled mixed call"
    );
}

#[test]
fn one_capacity_program_handles_changing_decode_counts_and_live_rows() {
    let Some((assets, buckets, ring)) = ordinary_assets() else {
        return;
    };
    let be = Arc::new(HsaBackend::new(0).expect("leased HSA device"));
    let mut engine = AmdEngine::load(
        be,
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .expect("load ordinary assets with runtime fusion");
    assert!(
        engine.batch() >= 4,
        "this test requires at least four slots"
    );
    let mut loaded_buckets: Vec<_> = (0..engine.n_programs())
        .filter_map(|program| engine.prefill_prog_t(program))
        .filter(|&rows| rows as usize > engine.batch())
        .collect();
    loaded_buckets.sort_unstable();
    loaded_buckets.dedup();
    assert_eq!(loaded_buckets, buckets, "ordinary bucket inventory");
    let mut failures = Vec::new();
    for rows in buckets {
        for decode_rows in 1..engine.batch() {
            assert_eq!(
                engine.mixed_step_rows(decode_rows, rows as usize - decode_rows),
                Some(rows),
                "runtime bucket {rows}, D{decode_rows}; requires PLOW_FUSION=1"
            );
        }
        for start in [0, ring - 64] {
            let mut decode_slots = vec![engine.batch() - 1, 1, 2];
            let mut decode_prompts = vec![prompt(97, 0), prompt(ring - 1, 3), prompt(257, 4)];
            let mut orders: Vec<&[usize]> =
                vec![&[0], &[2, 0, 1], &[1, 0], &[2], &[1, 2, 0], &[0, 1]];
            let mut takes = vec![
                1usize,
                1,
                2,
                4,
                129.min(rows as usize - 3),
                rows as usize - 2,
            ];
            if engine.batch() >= 8 {
                decode_slots.extend([5, 3, 6, 4]);
                decode_prompts.extend([
                    prompt(127, 5),
                    prompt(ring - 2, 6),
                    prompt(511, 7),
                    prompt(1023, 8),
                ]);
                orders.extend([
                    &[3, 2, 0, 1][..],
                    &[4, 0, 3, 2, 1],
                    &[6, 2, 4, 0, 5, 1, 3],
                    &[6, 3, 0],
                    &[5, 0, 4, 1, 6, 2],
                ]);
                takes.extend([1, 1, 1, rows as usize - 3, rows as usize - 6]);
            }
            let prefill_prompt = prompt(start + takes.iter().sum::<usize>() + 1, 8);
            assert!(
                prefill_prompt.len() <= engine.max_ctx(),
                "prefill ring-wrap context"
            );
            assert!(
                start + takes[..takes.len() - 1].iter().sum::<usize>() + rows as usize
                    <= engine.max_ctx(),
                "padded final mixed chunk context"
            );
            assert!(
                ring - 1 + orders.len() <= engine.max_ctx(),
                "decode ring-wrap context"
            );
            let first_prefill = engine.prefill_slot(0, &prefill_prompt).unwrap();
            let prefill_logits = logits(&engine);
            let (chunk_token, chunk_logits) =
                ordinary_prefill_control(&mut engine, rows, &prefill_prompt, start, &takes);
            assert_eq!(chunk_token, first_prefill, "ordinary chunked prefill token");
            report_whole_prefill_variance(
                &chunk_logits,
                &prefill_logits,
                &format!("capacity {rows} ordinary chunked vs whole prefill start {start}"),
            );
            let mut first = Vec::new();
            let mut reference = Vec::new();
            for tokens in &decode_prompts {
                let mut token = engine.prefill_slot(0, tokens).unwrap();
                first.push(token);
                let mut trajectory = Vec::new();
                for step in 0..orders.len() {
                    token = isolated_decode(&mut engine, token, (tokens.len() + step) as u32);
                    trajectory.push((token, logits(&engine)));
                }
                reference.push(trajectory);
            }
            let mixed_reference = capacity_decode_control(
                &mut engine,
                rows,
                start,
                &decode_prompts,
                &decode_slots,
                &orders,
                &takes,
                &vec![123; prefill_prompt.len()],
            );
            let unrelated_decode_prompts: Vec<_> = decode_prompts
                .iter()
                .map(|prompt| vec![123; prompt.len()])
                .collect();
            capacity_decode_control(
                &mut engine,
                rows,
                start,
                &unrelated_decode_prompts,
                &decode_slots,
                &orders,
                &takes,
                &prefill_prompt,
            );
            let isolated_prefill_token = finish_prefill(
                &mut engine,
                0,
                &prefill_prompt,
                prefill_prompt.len() as u32 - 1,
            );
            let isolated_prefill_logits = logits(&engine);
            let mut frontiers = vec![0u32; engine.batch()];
            for (i, &slot) in decode_slots.iter().enumerate() {
                assert_eq!(
                    engine.prefill_slot(slot, &decode_prompts[i]).unwrap(),
                    first[i]
                );
                frontiers[slot] = decode_prompts[i].len() as u32;
            }
            if start != 0 {
                engine.prefill_slot(0, &prefill_prompt[..start]).unwrap();
            }
            frontiers[0] = start as u32;
            let mut tokens = first;
            let mut advances = vec![0usize; decode_prompts.len()];
            for (round, (order, take)) in orders.iter().zip(takes).enumerate() {
                let decode: Vec<_> = order
                    .iter()
                    .map(|&i| DecodeRequest {
                        slot: decode_slots[i] as u32,
                        state_slot: decode_slots[i] as u32,
                        token: tokens[i],
                    })
                    .collect();
                let at = frontiers[0] as usize;
                let prefill = [PrefillRequest {
                    slot: 0,
                    state_slot: 0,
                    start: at as u32,
                    tokens: &prefill_prompt[at..at + take],
                    prompt_len: prefill_prompt.len() as u32,
                }];
                assert_eq!(
                    engine.mixed_step_rows(decode.len(), (rows as usize) - decode.len()),
                    Some(rows)
                );
                if round == 0 {
                    reject_bad_feeds(&mut engine, rows, &decode, &prefill, &frontiers);
                    let mut overfull = prefill[0];
                    overfull.tokens = &prefill_prompt[at..at + rows as usize];
                    let mut attempted = frontiers.clone();
                    let mut output = [u32::MAX];
                    let launches = engine.seg_launches;
                    assert!(engine
                        .mixed_step(rows, &decode, &[overfull], &mut attempted, &mut output)
                        .is_err());
                    assert_eq!(
                        engine.seg_launches, launches,
                        "row-capacity refusal before launch"
                    );
                    assert_eq!(attempted, frontiers);
                    assert_eq!(output, [u32::MAX]);
                    assert_eq!(
                        engine.mixed_step_rows(engine.batch(), 1),
                        None,
                        "decode capacity refusal"
                    );
                }
                let before = frontiers.clone();
                let launches = engine.seg_launches;
                let mut output = vec![u32::MAX; decode.len()];
                engine
                    .mixed_step(rows, &decode, &prefill, &mut frontiers, &mut output)
                    .unwrap();
                assert_eq!(
                    engine.seg_launches - launches,
                    1,
                    "one dynamic mixed launch"
                );
                assert_eq!(frontiers[0], before[0] + take as u32);
                for (i, &slot) in decode_slots.iter().enumerate() {
                    assert_eq!(
                        frontiers[slot],
                        before[slot] + u32::from(order.contains(&i)),
                        "parked slot {slot}"
                    );
                }
                let actual = logit_rows(&engine, decode.len());
                assert_eq!(
                    output, mixed_reference[round].0,
                    "same-shape mixed isolation tokens"
                );
                assert_eq!(
                    actual, mixed_reference[round].1,
                    "same-shape mixed isolation logits start {start}, round {round}"
                );
                eprintln!("capacity {rows}, start {start}, round {round}: same-shape mixed isolation exact");
                let vocab = actual.len() / decode.len();
                for (row, &i) in order.iter().enumerate() {
                    let expected = &reference[i][advances[i]];
                    let label = format!(
                        "capacity {rows}, start {start}, round {round}, live {}, D{}, slot {}",
                        decode.len() + take,
                        decode.len(),
                        decode_slots[i]
                    );
                    assert_eq!(output[row], expected.0, "{label}: greedy token");
                    compare_logits(
                        &actual[row * vocab..(row + 1) * vocab],
                        &expected.1,
                        &label,
                        &mut failures,
                    );
                    tokens[i] = output[row];
                    advances[i] += 1;
                }
            }
            assert_eq!(
                frontiers[0] as usize,
                prefill_prompt.len() - 1,
                "final sampling row stays isolated"
            );
            let actual_prefill_token =
                finish_prefill(&mut engine, 0, &prefill_prompt, frontiers[0]);
            assert_eq!(actual_prefill_token, first_prefill);
            assert_eq!(
                actual_prefill_token, isolated_prefill_token,
                "prefill isolation token"
            );
            assert!(
                logits(&engine) == isolated_prefill_logits,
                "prefill isolation logits"
            );
            eprintln!("capacity {rows}, start {start}: same-shape prefill isolation exact");
            report_whole_prefill_variance(
                &logits(&engine),
                &prefill_logits,
                &format!("capacity {rows} completed prefill start {start}"),
            );
            compare_logits(
                &logits(&engine),
                &chunk_logits,
                &format!("capacity {rows} mixed vs matched ordinary prefill start {start}"),
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn ordinary_prefill_control(
    engine: &mut AmdEngine,
    rows: u32,
    prompt: &[u32],
    start: usize,
    takes: &[usize],
) -> (u32, Vec<f32>) {
    if start != 0 {
        engine.prefill_slot(0, &prompt[..start]).unwrap();
    }
    let mut at = start as u32;
    for &take in takes {
        let mut step = engine.chunk_steps(&[rows], take as u32).unwrap()[0];
        step.c0 = at;
        engine.prefill_prepare(prompt, step).unwrap();
        engine.run_segmented(step.prog).unwrap();
        at += take as u32;
    }
    let token = finish_prefill(engine, 0, prompt, at);
    (token, logits(engine))
}

fn capacity_decode_control(
    engine: &mut AmdEngine,
    rows: u32,
    start: usize,
    prompts: &[Vec<u32>],
    slots: &[usize],
    orders: &[&[usize]],
    takes: &[usize],
    prefill_prompt: &[u32],
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut frontiers = vec![0; engine.batch()];
    let mut tokens: Vec<_> = prompts
        .iter()
        .zip(slots)
        .map(|(prompt, &slot)| {
            frontiers[slot] = prompt.len() as u32;
            engine.prefill_slot(slot, prompt).unwrap()
        })
        .collect();
    if start != 0 {
        engine.prefill_slot(0, &prefill_prompt[..start]).unwrap();
    }
    frontiers[0] = start as u32;
    let mut reference = Vec::new();
    for (order, &take) in orders.iter().zip(takes) {
        let decode: Vec<_> = order
            .iter()
            .map(|&i| DecodeRequest {
                slot: slots[i] as u32,
                state_slot: slots[i] as u32,
                token: tokens[i],
            })
            .collect();
        let at = frontiers[0] as usize;
        let prefill = [PrefillRequest {
            slot: 0,
            state_slot: 0,
            start: at as u32,
            tokens: &prefill_prompt[at..at + take],
            prompt_len: prefill_prompt.len() as u32,
        }];
        let mut output = vec![u32::MAX; decode.len()];
        let launches = engine.seg_launches;
        engine
            .mixed_step(rows, &decode, &prefill, &mut frontiers, &mut output)
            .unwrap();
        assert_eq!(engine.seg_launches - launches, 1);
        for (row, &i) in order.iter().enumerate() {
            tokens[i] = output[row];
        }
        reference.push((output, logit_rows(engine, decode.len())));
    }
    reference
}

mod dynamic_serving {
    use super::{ordinary_assets, prompt};
    use plowrt::device::{hsa::HsaBackend, Backend};
    use plowrt::exec::ExecutorSet;
    use plowrt::orch::Registry;
    use plowrt::serve::engine::{AmdServe, ServeEngine};
    use plowrt::serve::mux::{self, Job, MuxConfig};
    use plowrt::serve::stream::{self, StreamChunk};
    use plowrt::serve::{AppState, GenParams};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tracing_subscriber::prelude::*;

    struct MixedLaunches(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for MixedLaunches {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct Message(bool);
            impl tracing::field::Visit for Message {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}") == "AMD mixed prefill/decode launch";
                    }
                }
            }
            if event.metadata().target() == "plowrt::serve::mux" {
                let mut message = Message(false);
                event.record(&mut message);
                if message.0 {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }

    fn submit(state: &AppState, slug: &str, ids: Vec<u32>, limit: usize) -> stream::ChunkReceiver {
        let mut gen = GenParams {
            max_tokens: limit,
            ignore_eos: true,
            ..GenParams::default()
        };
        gen.params.temperature = 0.0;
        let (tx, rx) = stream::channel();
        state
            .mux(slug)
            .unwrap()
            .submit(Job {
                prompt_ids: ids,
                gen,
                arrived: Instant::now(),
                respond: tx,
            })
            .map_err(|_| ())
            .expect("submit");
        rx
    }

    async fn reply(mut rx: stream::ChunkReceiver, prompt_len: usize, limit: usize) -> Vec<u32> {
        tokio::time::timeout(Duration::from_secs(300), async move {
            let mut tokens = Vec::new();
            while let Some(chunk) = rx.recv().await {
                match chunk {
                    StreamChunk::Token { id, .. } => tokens.push(id),
                    StreamChunk::Done { usage, reason, .. } => {
                        assert_eq!(tokens.len(), limit);
                        assert_eq!(usage.prompt_tokens, prompt_len);
                        assert_eq!(usage.completion_tokens, limit);
                        assert_eq!(reason.as_str(), "length");
                        return tokens;
                    }
                    StreamChunk::Err(error) => panic!("stream failure: {error}"),
                }
            }
            panic!("stream ended without completion");
        })
        .await
        .expect("serving request timeout")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn capacity_mixed_mux_preserves_limits_cancellation_and_terminal_fallback() {
        let Some((assets, buckets, _)) = ordinary_assets() else {
            return;
        };
        let launches = Arc::new(AtomicUsize::new(0));
        // The model owns a separate OS thread, so a thread-local subscriber cannot observe it.
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(MixedLaunches(Arc::clone(&launches))),
        )
        .expect("this integration binary owns its tracing subscriber");
        let engine = AmdServe::load(
            &assets.join("model.pkt"),
            &assets.join("hsaco"),
            Some(&assets.join("checkpoint")),
        )
        .unwrap();
        assert!(
            engine.mixed_step_rows(1, 1).is_some(),
            "mixed serving must be enabled"
        );
        let max_ctx = engine.max_ctx();
        let capacity = *buckets.last().unwrap() as usize;
        let prompts = [
            prompt(97, 0),
            prompt((capacity * 3 + 1).min(max_ctx - 2), 1),
            vec![100],
        ];
        let limits = [16usize, 3, 1];
        let be: Arc<dyn Backend> = Arc::new(HsaBackend::new(0).unwrap());
        let execset = Arc::new(ExecutorSet::bringup(be).unwrap());
        let registry = Registry::new();
        let slug = registry.load(assets, None).unwrap();
        let state = Arc::new(AppState::new(registry, execset));
        state.install_gpu_engine(slug.clone(), ServeEngine::Amd(engine));
        let dispatcher = mux::spawn(
            slug.clone(),
            state.registry.get(&slug).unwrap(),
            Arc::clone(&state),
            MuxConfig {
                max_hold_ms: 0.0,
                slo_ms: 300_000.0,
                ..MuxConfig::default()
            },
        );
        state.install_mux(slug.clone(), dispatcher);
        let mut reference = Vec::new();
        for (ids, limit) in prompts.iter().zip(limits) {
            reference
                .push(reply(submit(&state, &slug, ids.clone(), limit), ids.len(), limit).await);
        }

        let mut decoding = submit(&state, &slug, prompts[0].clone(), 64);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(120), decoding.recv())
                .await
                .unwrap(),
            Some(StreamChunk::Token { .. })
        ));
        let before = launches.load(Ordering::SeqCst);
        let cancelled = submit(&state, &slug, prompts[1].clone(), 3);
        tokio::time::timeout(Duration::from_secs(120), async {
            while launches.load(Ordering::SeqCst) == before {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("a real mixed launch before prefill cancellation");
        drop(cancelled);
        drop(decoding);
        assert_eq!(
            reply(
                submit(&state, &slug, prompts[1].clone(), 3),
                prompts[1].len(),
                3
            )
            .await,
            reference[1],
            "reuse after mixed-prefill cancellation"
        );

        let pending: Vec<_> = prompts
            .iter()
            .zip(limits)
            .enumerate()
            .map(|(i, (ids, limit))| (i, submit(&state, &slug, ids.clone(), limit)))
            .collect();
        for (i, rx) in pending {
            assert_eq!(
                reply(rx, prompts[i].len(), limits[i]).await,
                reference[i],
                "concurrent case {i}, including one-token fallback"
            );
        }
        let mut rejected = submit(&state, &slug, vec![100; max_ctx], 1);
        assert!(
            matches!(
                tokio::time::timeout(Duration::from_secs(120), rejected.recv())
                    .await
                    .unwrap(),
                Some(StreamChunk::Err(_))
            ),
            "over-context request must fail"
        );
        assert_eq!(
            reply(submit(&state, &slug, prompts[2].clone(), 1), 1, 1).await,
            reference[2],
            "terminal fallback survives rejected context"
        );
        assert!(
            launches.load(Ordering::SeqCst) > before,
            "cannot pass by silently falling back for every request"
        );
        tokio::time::timeout(Duration::from_secs(120), state.mux(&slug).unwrap().drain())
            .await
            .expect("mux drain");
    }
}

#[test]
fn mixed_prefill_decode_matches_sequential_across_slots_and_ring_wrap() {
    let Some((assets, _, ring)) = ordinary_assets() else {
        return;
    };
    let be = Arc::new(HsaBackend::new(0).expect("leased HSA device"));
    let mut engine = AmdEngine::load(
        be,
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .expect("load mixed full-model assets");
    assert!(engine.batch() >= 4, "requires decode ladder through B4");
    assert!(engine.max_ctx() >= ring + 1024);
    let rows = engine
        .mixed_step_rows(1, 200)
        .expect("sampled mixed D1 variant");
    assert!(rows >= 201);
    let mut failures = Vec::new();
    parity_case(&mut engine, rows, 97, [0, 128], [801, 933], &mut failures);
    parity_case(
        &mut engine,
        rows,
        ring - 1,
        [ring - 64, ring + 64],
        [ring + 553, ring + 709],
        &mut failures,
    );
    multiple_decode_rows(&mut engine, &[3, 1], &[ring - 1, 127], &mut failures);
    multiple_decode_rows(
        &mut engine,
        &[2, 3, 1],
        &[127, ring - 1, 257],
        &mut failures,
    );
    assert!(
        failures.is_empty(),
        "logit parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "historical diagnostic: fixed T512/ring2048; whole-prefill variance is not the fusion gate"]
fn ordinary_ragged_prefill_matches_whole_prefill() {
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let be = Arc::new(HsaBackend::new(0).expect("leased HSA device"));
    let mut engine = AmdEngine::load(
        be,
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .unwrap();
    let mut failures = Vec::new();
    for (len, start, salt, chunks) in [
        (801, 0, 1, [53, 127, 31, 256]),
        (933, 128, 2, [91, 63, 99, 255]),
        (2601, 1984, 1, [53, 127, 31, 256]),
        (2757, 2112, 2, [91, 63, 99, 255]),
    ] {
        let tokens = prompt(len, salt);
        let expected = engine.prefill_slot(0, &tokens).unwrap();
        let reference = logits(&engine);
        let repeated = engine.prefill_slot(2, &tokens).unwrap();
        assert_eq!(repeated, expected);
        assert_eq!(logits(&engine), reference, "whole-prefill slot invariance");
        if start != 0 {
            engine.prefill_slot(0, &tokens[..start]).unwrap();
        }
        let mut from = start as u32;
        for count in chunks {
            let mut step = engine.chunk_steps(&[512], count).unwrap()[0];
            step.c0 = from;
            engine.prefill_prepare(&tokens, step).unwrap();
            engine.run_segmented(step.prog).unwrap();
            from += count;
        }
        let actual = finish_prefill(&mut engine, 0, &tokens, from);
        let chunk_reference = logits(&engine);
        let label = format!("ordinary chunks len {len}, start {start}, finish {from}");
        eprintln!("{label}: token {actual}, reference {expected}");
        assert_eq!(actual, expected, "{label}: greedy token");
        compare_logits(&chunk_reference, &reference, &label, &mut failures);

        let decode_prompt = prompt(if start < 1024 { 97 } else { 2047 }, 0);
        let mut token = engine.prefill_slot(3, &decode_prompt).unwrap();
        let mut frontiers = vec![0u32; engine.batch()];
        frontiers[3] = decode_prompt.len() as u32;
        if start != 0 {
            engine.prefill_slot(0, &tokens[..start]).unwrap();
        }
        frontiers[0] = start as u32;
        for count in chunks {
            let decode = [DecodeRequest {
                slot: 3,
                state_slot: 3,
                token,
            }];
            let at = frontiers[0];
            let prefill = [PrefillRequest {
                slot: 0,
                state_slot: 0,
                start: at,
                tokens: &tokens[at as usize..(at + count) as usize],
                prompt_len: len as u32,
            }];
            let mut output = [u32::MAX];
            engine
                .mixed_step(512, &decode, &prefill, &mut frontiers, &mut output)
                .unwrap();
            token = output[0];
        }
        assert_eq!(frontiers[0], from);
        let mixed = finish_prefill(&mut engine, 0, &tokens, from);
        let label = format!("mixed vs ordinary chunks len {len}, start {start}");
        eprintln!("{label}: token {mixed}, reference {actual}");
        assert_eq!(mixed, actual, "{label}: greedy token");
        compare_logits(&logits(&engine), &chunk_reference, &label, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "historical diagnostic: fixed T512, first53 KV rows; final-logit tolerance is not a KV gate"]
fn mixed_first_chunk_kv_matches_ordinary() {
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets.join("checkpoint/config.json")).unwrap())
            .unwrap();
    let config = &config["text_config"];
    assert_eq!(config["num_kv_shared_layers"].as_u64(), Some(0));
    let layout: Vec<_> = config["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .flat_map(|(layer, kind)| {
            let hd = config[if kind == "full_attention" {
                "global_head_dim"
            } else {
                "head_dim"
            }]
            .as_u64()
            .unwrap() as usize;
            [
                (format!("kv.{layer}.k"), 53 * hd * 2),
                (format!("kv.{layer}.v"), 53 * hd * 2),
            ]
        })
        .collect();
    let be = Arc::new(HsaBackend::new(0).expect("leased HSA device"));
    let mut engine = AmdEngine::load(
        be,
        &assets.join("model.pkt"),
        &assets.join("hsaco"),
        Some(&assets.join("checkpoint")),
    )
    .unwrap();
    let capture = |engine: &AmdEngine| -> Vec<Vec<u8>> {
        layout
            .iter()
            .map(|(name, bytes)| {
                let mut out = vec![0; *bytes];
                engine.read_tensor(name, &mut out).unwrap();
                out
            })
            .collect()
    };
    let tokens = prompt(801, 1);
    let step = engine.chunk_steps(&[512], 53).unwrap()[0];
    engine.prefill_prepare(&tokens, step).unwrap();
    engine.run_segmented(step.prog).unwrap();
    let reference = capture(&engine);
    engine.prefill_prepare(&tokens, step).unwrap();
    engine.run_segmented(step.prog).unwrap();
    assert_eq!(capture(&engine), reference, "ordinary KV repeat invariance");

    let decode_prompt = prompt(97, 0);
    let token = engine.prefill_slot(3, &decode_prompt).unwrap();
    let mut frontiers = vec![0u32; engine.batch()];
    frontiers[3] = decode_prompt.len() as u32;
    let decode = [DecodeRequest {
        slot: 3,
        state_slot: 3,
        token,
    }];
    let prefill = [PrefillRequest {
        slot: 0,
        state_slot: 0,
        start: 0,
        tokens: &tokens[..53],
        prompt_len: tokens.len() as u32,
    }];
    engine
        .mixed_step(512, &decode, &prefill, &mut frontiers, &mut [u32::MAX])
        .unwrap();
    let actual = capture(&engine);
    let mut failures = Vec::new();
    for (((name, _), actual), reference) in layout.iter().zip(actual).zip(reference) {
        let mismatch = actual
            .chunks_exact(2)
            .zip(reference.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "{name} head0 first53rows: BF16 mismatches={mismatch}/{}",
            actual.len() / 2
        );
        let decode = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()
        };
        compare_logits(&decode(&actual), &decode(&reference), name, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn prompt(len: usize, salt: usize) -> Vec<u32> {
    (0..len)
        .map(|i| 100 + ((i + salt * 7) % 100) as u32)
        .collect()
}

fn logits(engine: &AmdEngine) -> Vec<f32> {
    logit_rows(engine, 1)
}

fn logit_rows(engine: &AmdEngine, rows: usize) -> Vec<f32> {
    let vocab_bytes = engine.tensor_bytes("act.logits").expect("logits") as usize / engine.batch();
    let mut bytes = vec![0; rows * vocab_bytes];
    engine.read_tensor("act.logits", &mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

fn compare_logits(actual: &[f32], reference: &[f32], label: &str, failures: &mut Vec<String>) {
    assert_eq!(actual.len(), reference.len());
    let (mut error, mut scale, mut max_abs) = (0.0f64, 0.0f64, 0.0f32);
    for (&a, &b) in actual.iter().zip(reference) {
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite logits");
        let delta = a - b;
        error += f64::from(delta).powi(2);
        scale += f64::from(b).powi(2);
        max_abs = max_abs.max(delta.abs());
    }
    let relative_l2 = (error / scale.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("{label}: logit relative_l2={relative_l2:.6}, max_abs={max_abs:.6}");
    // The mixed matrix tile changes BF16 accumulation, but not the model or sampling.
    if relative_l2 > 0.01 || max_abs > 0.5 {
        failures.push(format!(
            "{label}: relative_l2={relative_l2:.6}, max_abs={max_abs:.6}"
        ));
    }
}

fn report_whole_prefill_variance(actual: &[f32], reference: &[f32], label: &str) {
    let mut differences = Vec::new();
    compare_logits(actual, reference, label, &mut differences);
    for difference in differences {
        eprintln!("whole-prefill chunk-boundary diagnostic: {difference}");
    }
}

fn isolated_decode(engine: &mut AmdEngine, token: u32, position: u32) -> u32 {
    let mut ids = vec![0u32; engine.batch()];
    ids[0] = token;
    engine
        .write_tensor("in.ids", bytemuck::cast_slice(&ids))
        .unwrap();
    let mut pos = vec![0; engine.batch()];
    let mut kvlen = vec![1; engine.batch()];
    pos[0] = position;
    kvlen[0] = position + 1;
    let dp = engine.decode_prog_for(1);
    engine.decode_step_batched_at(&pos, &kvlen, dp).unwrap()[0]
}

fn finish_prefill(engine: &mut AmdEngine, slot: usize, tokens: &[u32], from: u32) -> u32 {
    let chunks = engine.plan_for(tokens.len() as u32 - from).unwrap();
    let steps = engine
        .chunk_steps_from(&chunks, from, tokens.len() as u32)
        .unwrap();
    engine.kv_rebase(slot).unwrap();
    for step in steps {
        engine.prefill_prepare(tokens, step).unwrap();
        engine.run_segmented(step.prog).unwrap();
    }
    engine.kv_rebase(0).unwrap();
    engine.read_sampled().unwrap()
}

fn parity_case(
    engine: &mut AmdEngine,
    rows: u32,
    decode_len: usize,
    starts: [usize; 2],
    lengths: [usize; 2],
    failures: &mut Vec<String>,
) {
    let decode_prompt = prompt(decode_len, 0);
    let prompts = [prompt(lengths[0], 1), prompt(lengths[1], 2)];
    let first = engine.prefill_slot(0, &decode_prompt).unwrap();
    let mut token = first;
    let mut decode_reference = Vec::new();
    for step in 0..4 {
        token = isolated_decode(engine, token, (decode_len + step) as u32);
        decode_reference.push((token, logits(engine)));
    }
    let mut takes = [[53, 91], [127, 63], [31, 99], [113, 71]];
    let available: [usize; 2] = std::array::from_fn(|i| {
        lengths[i] - starts[i] - takes[..3].iter().map(|take| take[i]).sum::<usize>() - 1
    });
    let capacity = rows as usize - 1;
    if available.iter().sum::<usize>() >= capacity {
        takes[3][1] = (capacity / 2).min(available[1]);
        takes[3][0] = (capacity - takes[3][1]).min(available[0]);
        takes[3][1] = capacity - takes[3][0];
    }
    let mixed_reference = capacity_decode_control(
        engine,
        rows,
        0,
        std::slice::from_ref(&decode_prompt),
        &[3],
        &[&[0][..]; 4],
        &takes.map(|take| take.iter().sum()),
        &vec![123; takes.iter().flatten().sum::<usize>() + 1],
    );
    let prefill_reference: Vec<_> = prompts
        .iter()
        .map(|p| {
            let token = engine.prefill_slot(0, p).unwrap();
            (token, logits(engine))
        })
        .collect();
    let chunk_reference: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let reference = ordinary_prefill_control(
                engine,
                rows,
                prompt,
                starts[i],
                &takes.map(|take| take[i]),
            );
            assert_eq!(
                reference.0, prefill_reference[i].0,
                "ordinary chunked prefill token"
            );
            report_whole_prefill_variance(
                &reference.1,
                &prefill_reference[i].1,
                &format!("ordinary chunked vs whole prefill {i}, start {}", starts[i]),
            );
            reference
        })
        .collect();

    assert_eq!(engine.prefill_slot(3, &decode_prompt).unwrap(), first);
    let slots = [0usize, 2];
    let mut frontiers = vec![0u32; engine.batch()];
    frontiers[3] = decode_len as u32;
    for i in 0..2 {
        if starts[i] != 0 {
            engine
                .prefill_slot(slots[i], &prompts[i][..starts[i]])
                .unwrap();
        }
        frontiers[slots[i]] = starts[i] as u32;
    }
    let mut output = [u32::MAX];
    token = first;
    for (round, take) in takes.into_iter().enumerate() {
        let decode = [DecodeRequest {
            slot: 3,
            state_slot: 3,
            token,
        }];
        let prefill: Vec<_> = (0..2)
            .map(|i| {
                let start = frontiers[slots[i]];
                PrefillRequest {
                    slot: slots[i] as u32,
                    state_slot: slots[i] as u32,
                    start,
                    tokens: &prompts[i][start as usize..start as usize + take[i]],
                    prompt_len: prompts[i].len() as u32,
                }
            })
            .collect();
        if round == 0 {
            reject_bad_feeds(engine, rows, &decode, &prefill, &frontiers);
        }
        let before = frontiers.clone();
        let launches = engine.seg_launches;
        engine
            .mixed_step(rows, &decode, &prefill, &mut frontiers, &mut output)
            .unwrap();
        assert_eq!(engine.seg_launches - launches, 1, "one mixed launch");
        assert_eq!(frontiers[3], before[3] + 1);
        assert_eq!(frontiers[1], before[1], "parked physical slot");
        for i in 0..2 {
            assert_eq!(frontiers[slots[i]], before[slots[i]] + take[i] as u32);
        }
        assert_eq!(
            output[0], decode_reference[round].0,
            "decode {decode_len}, round {round}"
        );
        let actual_logits = logits(engine);
        assert_eq!(
            output.as_slice(),
            mixed_reference[round].0,
            "mixed arithmetic isolation token"
        );
        assert!(
            actual_logits == mixed_reference[round].1,
            "mixed arithmetic isolation logits"
        );
        eprintln!("decode {decode_len}, round {round}: mixed isolation exact");
        compare_logits(
            &actual_logits,
            &decode_reference[round].1,
            &format!("decode {decode_len}, round {round}"),
            failures,
        );
        token = output[0];
    }
    for i in 0..2 {
        let actual = finish_prefill(engine, slots[i], &prompts[i], frontiers[slots[i]]);
        assert_eq!(actual, prefill_reference[i].0, "prefill slot {}", slots[i]);
        report_whole_prefill_variance(
            &logits(engine),
            &prefill_reference[i].1,
            &format!("prefill slot {}, start {}", slots[i], starts[i]),
        );
        compare_logits(
            &logits(engine),
            &chunk_reference[i].1,
            &format!(
                "mixed vs matched ordinary prefill slot {}, start {}",
                slots[i], starts[i]
            ),
            failures,
        );
    }
}

fn multiple_decode_rows(
    engine: &mut AmdEngine,
    slots: &[usize],
    lengths: &[usize],
    failures: &mut Vec<String>,
) {
    const STEPS: usize = 3;
    let rows = engine
        .mixed_step_rows(slots.len(), 256)
        .expect("sampled mixed D2/D3 variant");
    let capacity = rows as usize - slots.len();
    assert!(capacity >= 128);
    let prompts: Vec<_> = lengths
        .iter()
        .enumerate()
        .map(|(i, &n)| prompt(n, i + 3))
        .collect();
    let mut first = Vec::new();
    let mut reference = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let mut token = engine.prefill_slot(0, p).unwrap();
        first.push(token);
        let mut trajectory = Vec::new();
        for step in 0..STEPS {
            token = isolated_decode(engine, token, (lengths[i] + step) as u32);
            trajectory.push((token, logits(engine)));
        }
        reference.push(trajectory);
    }
    let prefill_prompt = prompt(capacity + 401, 7);
    let prefill_token = engine.prefill_slot(0, &prefill_prompt).unwrap();
    let prefill_logits = logits(engine);
    let chunk_reference =
        ordinary_prefill_control(engine, rows, &prefill_prompt, 0, &[96, 127, capacity]);
    assert_eq!(chunk_reference.0, prefill_token);
    report_whole_prefill_variance(
        &chunk_reference.1,
        &prefill_logits,
        &format!("D{} ordinary chunked vs whole prefill", slots.len()),
    );
    let mut frontiers = vec![0u32; engine.batch()];
    for (i, &slot) in slots.iter().enumerate() {
        assert_eq!(engine.prefill_slot(slot, &prompts[i]).unwrap(), first[i]);
        frontiers[slot] = lengths[i] as u32;
    }
    let mut tokens = first;
    let mut output = vec![u32::MAX; slots.len()];
    for (round, take) in [96, 127, capacity].into_iter().enumerate() {
        let order: Vec<_> = (0..slots.len())
            .map(|row| (row + round) % slots.len())
            .collect();
        let decode: Vec<_> = order
            .iter()
            .map(|&i| DecodeRequest {
                slot: slots[i] as u32,
                state_slot: slots[i] as u32,
                token: tokens[i],
            })
            .collect();
        let start = frontiers[0];
        let prefill = [PrefillRequest {
            slot: 0,
            state_slot: 0,
            start,
            tokens: &prefill_prompt[start as usize..start as usize + take],
            prompt_len: prefill_prompt.len() as u32,
        }];
        if round == 0 {
            let mut duplicate = decode.clone();
            duplicate[1] = duplicate[0];
            let mut attempted = frontiers.clone();
            let launches = engine.seg_launches;
            assert!(engine
                .mixed_step(rows, &duplicate, &prefill, &mut attempted, &mut output)
                .is_err());
            assert_eq!(attempted, frontiers);
            assert!(output.iter().all(|&token| token == u32::MAX));
            assert_eq!(engine.seg_launches, launches);
        }
        let before = frontiers.clone();
        let launches = engine.seg_launches;
        engine
            .mixed_step(rows, &decode, &prefill, &mut frontiers, &mut output)
            .unwrap();
        assert_eq!(
            engine.seg_launches - launches,
            1,
            "D{} mixed launch",
            slots.len()
        );
        for slot in 0..engine.batch() {
            let advanced = if slot == 0 {
                take as u32
            } else {
                u32::from(slots.contains(&slot))
            };
            assert_eq!(
                frontiers[slot],
                before[slot] + advanced,
                "frontier slot {slot}"
            );
        }
        let actual_logits = logit_rows(engine, slots.len());
        let vocab = actual_logits.len() / slots.len();
        for (row, &i) in order.iter().enumerate() {
            assert_eq!(
                output[row],
                reference[i][round].0,
                "D{} row {row}, slot {}, round {round}",
                slots.len(),
                slots[i]
            );
            compare_logits(
                &actual_logits[row * vocab..(row + 1) * vocab],
                &reference[i][round].1,
                &format!(
                    "D{} row {row}, slot {}, round {round}",
                    slots.len(),
                    slots[i]
                ),
                failures,
            );
            tokens[i] = output[row];
        }
    }
    assert_eq!(
        finish_prefill(engine, 0, &prefill_prompt, frontiers[0]),
        prefill_token
    );
    report_whole_prefill_variance(
        &logits(engine),
        &prefill_logits,
        &format!("D{} completed prefill", slots.len()),
    );
    compare_logits(
        &logits(engine),
        &chunk_reference.1,
        &format!("D{} mixed vs matched ordinary prefill", slots.len()),
        failures,
    );
}

fn reject_bad_feeds(
    engine: &mut AmdEngine,
    rows: u32,
    decode: &[DecodeRequest],
    prefill: &[PrefillRequest<'_>],
    frontiers: &[u32],
) {
    let mut check = |decode: &[DecodeRequest], prefill: &[PrefillRequest<'_>], initial: &[u32]| {
        let mut attempted = initial.to_vec();
        let mut output = [u32::MAX];
        let launches = engine.seg_launches;
        assert!(engine
            .mixed_step(rows, decode, prefill, &mut attempted, &mut output)
            .is_err());
        assert_eq!(engine.seg_launches, launches, "reject before launch");
        assert_eq!(attempted, initial, "rejected frontiers must be unchanged");
        assert_eq!(output, [u32::MAX], "rejected output must be unchanged");
    };
    check(decode, &[prefill[0], prefill[0]], frontiers);
    check(&[decode[0], decode[0]], prefill, frontiers);
    let mut invalid = prefill[0];
    invalid.start += 1;
    check(decode, &[invalid], frontiers);
    invalid = prefill[0];
    invalid.tokens = &[];
    check(decode, &[invalid], frontiers);
    invalid = prefill[0];
    invalid.slot = decode[0].slot;
    invalid.state_slot = decode[0].state_slot;
    check(decode, &[invalid], frontiers);
    invalid = prefill[0];
    invalid.slot = frontiers.len() as u32;
    check(decode, &[invalid], frontiers);
    let mut invalid_decode = decode[0];
    invalid_decode.token = u32::MAX;
    check(&[invalid_decode], prefill, frontiers);
    invalid = prefill[0];
    invalid.tokens = &[u32::MAX];
    check(decode, &[invalid], frontiers);
    let mut end = frontiers.to_vec();
    end[decode[0].slot as usize] = engine.max_ctx() as u32;
    let mut output = [u32::MAX];
    let before = end.clone();
    assert!(engine
        .mixed_step(rows, decode, prefill, &mut end, &mut output)
        .is_err());
    assert_eq!(end, before);
    assert_eq!(output, [u32::MAX]);
}
