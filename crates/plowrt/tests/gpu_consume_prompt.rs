//! Decode-loop prompt consumption (`GpuEngine::consume_prompt`) must produce
//! EXACTLY the first token and follow-on greedy tokens that per-token
//! `step_slots` produces from the same prompt. One host sync vs L must not
//! change numerics.
//!
//! Gated on `PLOW_GPU_TEST=1` + assets (`PLOW_GPU_ASSETS`). Skips silently.

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::exec::gpu::GpuEngine;

fn generate_from_prompt(e: &mut GpuEngine, prompt: &[u32], n_gen: usize, fused: bool) -> Vec<u32> {
    e.begin_slot(0, prompt.len() + n_gen + 1).expect("begin");
    let mut toks = Vec::new();
    let first = if fused {
        e.consume_prompt(0, prompt, &mut toks)
            .expect("consume_prompt")
    } else {
        let mut t = 0u32;
        for &id in prompt {
            e.step_slots(&[(0, id)], &mut toks).expect("step (prompt)");
            t = toks[0];
        }
        t
    };
    let mut out = vec![first];
    let mut last = first;
    for _ in 1..n_gen {
        e.step_slots(&[(0, last)], &mut toks).expect("step (gen)");
        last = toks[0];
        out.push(last);
    }
    out
}

#[test]
fn consume_prompt_matches_step_slots_greedy() {
    let _env = common::env_guard();
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    let ckpt = assets.join("checkpoint");

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("engine load");

    let prompt: Vec<u32> = (0..48u32).map(|i| 100 + i).collect();
    const N_GEN: usize = 8;

    let single = generate_from_prompt(&mut e, &prompt, N_GEN, false);
    let fused = generate_from_prompt(&mut e, &prompt, N_GEN, true);

    assert_eq!(
        fused, single,
        "consume_prompt diverged from per-token step_slots:\n fused={fused:?}\nsingle={single:?}"
    );
    eprintln!(
        "consume_prompt OK: {} prompt + {N_GEN} gen tokens identical; first {:?}",
        prompt.len(),
        &single[..single.len().min(8)]
    );
}

struct LogitSnapshot {
    token: u32,
    bits: Vec<u32>,
}

fn snapshot(e: &mut GpuEngine, row: usize, token: u32) -> LogitSnapshot {
    let mut logits = Vec::new();
    e.logits_row(row, &mut logits).expect("full logits");
    assert!(!logits.is_empty() && logits.iter().all(|v| v.is_finite()));
    LogitSnapshot {
        token,
        bits: logits.into_iter().map(f32::to_bits).collect(),
    }
}

fn compare_snapshot(actual: LogitSnapshot, expected: &LogitSnapshot, case: &str) {
    assert_eq!(actual.token, expected.token, "{case}: greedy token");
    assert_eq!(actual.bits.len(), expected.bits.len(), "{case}: vocabulary");
    if let Some((i, (&got, &want))) = actual
        .bits
        .iter()
        .zip(&expected.bits)
        .enumerate()
        .find(|(_, (got, want))| got != want)
    {
        panic!("{case}: logit {i} differs: got {got:#010x}, expected {want:#010x}");
    }
}

#[test]
fn serialized_tma_slots_match_isolated_full_logits() {
    serialized_tma_slot_parity(false);
}

#[test]
fn live_tma_all_slots_match_flat_full_logits() {
    serialized_tma_slot_parity(true);
}

fn serialized_tma_slot_parity(all_slots_live: bool) {
    let _env = common::env_guard();
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + B>=2 TMA assets)");
        return;
    }
    let _config = common::EnvScope::set(&[
        ("PLOW_VMM_LIVE", "0"),
        ("PLOW_VMM_PREFIX", "0"),
        ("PLOW_PREFIX_CACHE", "0"),
        ("PLOW_PF_BATCH", "0"),
        ("PLOW_VMM_BLOCK_MIB", "2"),
        ("PLOW_KV_POOL_MIB", "0"),
    ]);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "plowrt=info".into()))
        .try_init();
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let packet = plowrt::asset::devblob::DevBlob::find_in_dir(&assets)
        .expect("find packet")
        .expect("packet required");
    let blob = plowrt::asset::devblob::DevBlob::parse(&std::fs::read(packet).unwrap())
        .expect("parse packet");
    let referenced_maps = blob
        .prefill_progs()
        .iter()
        .flat_map(|p| &p.insts)
        .filter(|inst| inst.op == packet::dev::DevOp::FlashPrefill as u16)
        .filter(|inst| {
            inst.t[6] == packet::dev::TENSOR_NONE16
                && blob.gen.iter().any(|g| {
                    g.kind == packet::rope::GEN_TMAP_KV_PAIR && g.tensor == u32::from(inst.t[7])
                })
        })
        .count();
    assert!(
        referenced_maps > 0,
        "packet requires serialized KV TMA consumers"
    );
    assert!(
        blob.prefill_progs().iter().all(|p| p.t % 32 == 0),
        "use prefill buckets with partial padding for the +1-token cases"
    );
    let short_rows = blob.prefill_progs().iter().map(|p| p.t).min().unwrap() as usize;
    if all_slots_live {
        plowrt::memory::vmm::LiveKvLayout::from_blob(&blob).expect("live packet geometry");
    }
    drop(blob);

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut e =
        GpuEngine::load(Arc::clone(&be), &assets, &assets.join("checkpoint")).expect("engine load");
    assert!(e.batch() >= 2 && e.has_prefill() && e.max_ctx() >= 16400);
    assert!(e.vmm_stats().is_none(), "gate requires flat KV allocation");
    let slots = if all_slots_live { e.batch() } else { 2 };
    let prompts: Vec<Vec<u32>> = [8192, 16384, 8193, 16385]
        .into_iter()
        .chain(std::iter::repeat_n(short_rows, slots - 2))
        .enumerate()
        .map(|(case, len)| {
            (0..len)
                .map(|i| 100 + ((i * (2 * case + 1) + case * 173) % 1000) as u32)
                .collect()
        })
        .collect();
    let mut reference = Vec::new();
    let mut decoded = Vec::new();
    for (case, prompt) in prompts.iter().enumerate() {
        let decode_steps = if case < 4 { 8 } else { 12 };
        e.begin_slot(0, prompt.len() + decode_steps + 1)
            .expect("baseline reset");
        let mut token = e.prefill_slot(0, prompt).expect("baseline prefill");
        let mut trajectory = vec![snapshot(&mut e, 0, token)];
        for _ in 0..decode_steps {
            e.step_slots(&[(0, token)], &mut decoded)
                .expect("baseline decode");
            token = decoded[0];
            trajectory.push(snapshot(&mut e, 0, token));
        }
        reference.push(trajectory);
        eprintln!(
            "isolated case={case} prompt={} full-logit snapshots={}",
            prompt.len(),
            decode_steps + 1
        );
    }
    for live in [false, true]
        .into_iter()
        .take(if all_slots_live { 2 } else { 1 })
    {
        if live {
            drop(e);
            std::env::set_var("PLOW_VMM_LIVE", "1");
            e = GpuEngine::load(Arc::clone(&be), &assets, &assets.join("checkpoint"))
                .expect("live engine load");
            assert_eq!(e.batch(), slots);
        }
        for slot in 0..e.batch() {
            e.begin_slot(slot, 1).expect("clear baseline positions");
        }

        let mut active: Vec<usize> = (0..slots)
            .map(|slot| if slot < 2 { slot } else { slot + 2 })
            .collect();
        let mut steps = vec![0usize; slots];
        let mut tokens = vec![0u32; slots];
        for slot in 0..slots {
            let case = active[slot];
            e.begin_slot(slot, prompts[case].len() + 13)
                .expect("candidate reset");
            tokens[slot] = e
                .prefill_slot(slot, &prompts[case])
                .expect("candidate prefill");
            // Serialized prefill's M=1 head always writes row 0, regardless of slot.
            compare_snapshot(
                snapshot(&mut e, 0, tokens[slot]),
                &reference[case][0],
                &format!("live={live} initial slot={slot} prefill"),
            );
        }
        for phase in 0..3 {
            if phase > 0 {
                let slot = if phase == 1 { 1 } else { 0 };
                let case = phase + 1;
                active[slot] = case;
                steps[slot] = 0;
                e.begin_slot(slot, prompts[case].len() + 9)
                    .expect("interleaved reset");
                tokens[slot] = e.prefill_slot(slot, &prompts[case]).expect("reset prefill");
                compare_snapshot(
                    snapshot(&mut e, 0, tokens[slot]),
                    &reference[case][0],
                    &format!("live={live} reset slot={slot} case={case} prefill"),
                );
            }
            for _ in 0..4 {
                let inputs: Vec<_> = tokens.iter().copied().enumerate().collect();
                e.step_slots(&inputs, &mut decoded)
                    .expect("interleaved decode");
                for slot in 0..slots {
                    tokens[slot] = decoded[slot];
                    steps[slot] += 1;
                    let case = active[slot];
                    compare_snapshot(
                        snapshot(&mut e, slot, tokens[slot]),
                        &reference[case][steps[slot]],
                        &format!("live={live} slot={slot} case={case} decode={}", steps[slot]),
                    );
                }
            }
            for slot in 0..e.batch() {
                assert_eq!(e.attached_rows(slot), 0);
            }
            if live {
                let stats = e.vmm_stats().expect("live allocator enabled");
                assert_eq!(stats.attach_hits + stats.attach_misses, 0);
                assert_eq!(stats.tokens_attached + stats.blocks_shared_mapped, 0);
                assert_eq!(
                    stats.cache_blocks + stats.blocks_pooled + stats.blocks_reused,
                    0
                );
                assert!(stats.blocks_live > 0);
                eprintln!("live phase={phase} stats={stats:?}");
            } else {
                assert!(e.vmm_stats().is_none());
            }
            eprintln!("live={live} interleaved phase={phase} cases={active:?} steps={steps:?}: full logits exact");
        }
        for slot in 0..e.batch() {
            e.begin_slot(slot, 1).expect("release active slots");
        }
    }
}
