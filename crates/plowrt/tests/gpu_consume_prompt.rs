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
use std::time::Instant;

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
#[ignore = "requires free H100, PLOW_GPU_TEST=1 and B16 FP8-KV PLOW_GPU_ASSETS"]
fn fp8_live_allocations_match_prefix_reference_across_rungs_and_slot_reuse() {
    let _env = common::env_guard();
    assert_eq!(std::env::var("PLOW_GPU_TEST").as_deref(), Ok("1"));
    let _config = common::EnvScope::set(&[
        ("PLOW_VMM_LIVE", "0"),
        ("PLOW_VMM_LIVE_RINGS", "0"),
        ("PLOW_VMM_PREFIX", "1"),
        ("PLOW_PREFIX_CACHE", "1"),
        ("PLOW_TOKEN_BATCH", "0"),
        ("PLOW_PF_BATCH", "0"),
        ("PLOW_MULTISTEP", "0"),
        ("PLOW_KV_POOL_MIB", "0"),
    ]);
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let packet = plowrt::asset::devblob::DevBlob::find_in_dir(&assets)
        .unwrap()
        .unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&std::fs::read(packet).unwrap()).unwrap();
    let manifest = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
    assert!(manifest.caches.iter().all(|c| c.scales.is_some()));
    plowrt::memory::vmm::LiveKvLayout::from_manifest(&blob, &manifest).unwrap();
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be.clone(), &assets, &assets.join("checkpoint")).unwrap();
    assert_eq!(e.batch(), 16);
    assert_eq!(&*e.effective_decode_rungs(), [1, 2, 4, 8, 16]);
    assert!(e.vmm_stats().is_some());
    let prompts: Vec<Vec<u32>> = (0..16)
        .map(|slot| {
            let len = match slot {
                0 => 1025,
                15 => 2051,
                _ => 129 + 17 * (slot % 3),
            };
            (0..len)
                .map(|i| 100 + ((i * (2 * slot + 1) + slot * 173) % 1000) as u32)
                .collect()
        })
        .collect();
    let mut reference = Vec::new();
    let mut decoded = Vec::new();
    for prompt in &prompts {
        e.begin_slot(0, prompt.len() + 7).unwrap();
        let mut token = e.prefill_slot(0, prompt).unwrap();
        let mut trajectory = vec![snapshot(&mut e, 0, token)];
        for _ in 0..6 {
            e.step_slots(&[(0, token)], &mut decoded).unwrap();
            token = decoded[0];
            trajectory.push(snapshot(&mut e, 0, token));
        }
        reference.push(trajectory);
    }
    drop(e);
    for lazy_scales in [false, true] {
        std::env::set_var("PLOW_VMM_LIVE", "1");
        std::env::set_var("PLOW_VMM_LIVE_RINGS", if lazy_scales { "1" } else { "0" });
        std::env::set_var("PLOW_VMM_PREFIX", "0");
        std::env::set_var("PLOW_PREFIX_CACHE", "0");
        let mut e = GpuEngine::load(be.clone(), &assets, &assets.join("checkpoint")).unwrap();
        assert_eq!(&*e.effective_decode_rungs(), [1, 2, 4, 8, 16]);
        assert!(e.vmm_stats().is_some());
        assert_eq!(e.live_ring_stats().is_some(), lazy_scales);
        let mut tokens = [0; 16];
        let mut steps = [0; 16];
        let mut active = 0;
        for width in [1, 2, 4, 8, 16] {
            for slot in active..width {
                e.begin_slot(slot, prompts[slot].len() + 7).unwrap();
                tokens[slot] = e.prefill_slot(slot, &prompts[slot]).unwrap();
                compare_snapshot(
                    snapshot(&mut e, 0, tokens[slot]),
                    &reference[slot][0],
                    &format!("lazy={lazy_scales} slot={slot} prefill"),
                );
            }
            active = width;
            let inputs: Vec<_> = tokens[..width].iter().copied().enumerate().collect();
            e.step_slots(&inputs, &mut decoded).unwrap();
            for slot in 0..width {
                tokens[slot] = decoded[slot];
                steps[slot] += 1;
                compare_snapshot(
                    snapshot(&mut e, slot, tokens[slot]),
                    &reference[slot][steps[slot]],
                    &format!("lazy={lazy_scales} rung={width} slot={slot}"),
                );
            }
        }
        e.begin_slot(15, prompts[0].len() + 7).unwrap();
        tokens[15] = e.prefill_slot(15, &prompts[0]).unwrap();
        compare_snapshot(
            snapshot(&mut e, 0, tokens[15]),
            &reference[0][0],
            "reused slot prefill",
        );
        let inputs: Vec<_> = tokens.iter().copied().enumerate().collect();
        e.step_slots(&inputs, &mut decoded).unwrap();
        for slot in 0..16 {
            let (case, step) = if slot == 15 {
                (0, 1)
            } else {
                (slot, steps[slot] + 1)
            };
            compare_snapshot(
                snapshot(&mut e, slot, decoded[slot]),
                &reference[case][step],
                "reused slot decode",
            );
        }
        assert_eq!(e.vmm_stats().unwrap().attach_hits, 0);
        eprintln!("FP8 LIVE lazy_scales={lazy_scales}: 64 exact full-vocabulary frames, all rungs and reused slot passed");
    }
}

#[test]
fn serialized_tma_slots_match_isolated_full_logits() {
    serialized_tma_slot_parity(false, false);
}

#[test]
fn live_tma_all_slots_match_flat_full_logits() {
    serialized_tma_slot_parity(true, false);
}

#[test]
fn live_ring_prefix_all_slots_match_live_full_logits() {
    serialized_tma_slot_parity(true, true);
}

fn serialized_tma_slot_parity(all_slots_live: bool, lazy_rings: bool) {
    let _env = common::env_guard();
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + B>=2 TMA assets)");
        return;
    }
    let mut settings = vec![
        ("PLOW_VMM_LIVE", if lazy_rings { "1" } else { "0" }),
        ("PLOW_VMM_LIVE_RINGS", "0"),
        ("PLOW_VMM_PREFIX", "0"),
        ("PLOW_PREFIX_CACHE", "0"),
        ("PLOW_PF_BATCH", "0"),
        ("PLOW_VMM_BLOCK_MIB", "2"),
        ("PLOW_KV_POOL_MIB", "0"),
    ];
    if !lazy_rings {
        settings.push(("PLOW_MULTISTEP", "0"));
    }
    let _config = common::EnvScope::set(&settings);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "plowrt=info".into()))
        .try_init();
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    let packet = plowrt::asset::devblob::DevBlob::find_in_dir(&assets)
        .expect("find packet")
        .expect("packet required");
    let raw = std::fs::read(packet).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).expect("parse packet");
    let packet_live = blob
        .sections
        .iter()
        .any(|section| section.name == plow_asset::packed_prefill::SECTION)
        && plowrt::memory::vmm::LiveKvLayout::manifest(&blob, &raw)
            .expect("live manifest")
            .is_some_and(|manifest| manifest.caches.iter().any(|cache| cache.window == 0));
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
    assert_eq!(e.vmm_stats().is_some(), lazy_rings || packet_live);
    assert!(
        e.live_ring_stats().is_none(),
        "baseline rings must stay flat"
    );
    if lazy_rings {
        assert_eq!(e.batch(), 16, "cold-prefix gate requires B16 ladder");
    }
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
        let effective_live = live || packet_live;
        if live {
            drop(e);
            std::env::set_var("PLOW_VMM_LIVE", "1");
            std::env::set_var("PLOW_VMM_LIVE_RINGS", if lazy_rings { "1" } else { "0" });
            e = GpuEngine::load(Arc::clone(&be), &assets, &assets.join("checkpoint"))
                .expect("live engine load");
            assert_eq!(e.batch(), slots);
        }
        if live && lazy_rings {
            assert_eq!(e.live_ring_stats().unwrap().mapped_slots, 0);
            for (case, slot, prefill_slots, prefix) in [(0, 0, 1, 1), (1, 3, 2, 4), (2, 15, 5, 16)]
            {
                let prompt = &prompts[case];
                let start = Instant::now();
                e.begin_slot(slot, prompt.len() + 13)
                    .expect("cold slot map");
                eprintln!(
                    "cold slot={slot} begin_ms={} rings={:?}",
                    start.elapsed().as_secs_f64() * 1e3,
                    e.live_ring_stats()
                );
                let mut token = e.prefill_slot(slot, prompt).expect("cold slot prefill");
                compare_snapshot(
                    snapshot(&mut e, 0, token),
                    &reference[case][0],
                    &format!("cold slot={slot} prefill"),
                );
                assert_eq!(e.live_ring_stats().unwrap().mapped_slots, prefill_slots);
                e.step_slots(&[(slot, token)], &mut decoded)
                    .expect("cold rung decode");
                token = decoded[0];
                compare_snapshot(
                    snapshot(&mut e, slot, token),
                    &reference[case][1],
                    &format!("cold slot={slot} decode"),
                );
                let stats = e.live_ring_stats().unwrap();
                assert_eq!((stats.mapped_slots, stats.mapped_prefix), (prefix, prefix));
                token = e
                    .consume_prompt(slot, &[token], &mut decoded)
                    .expect("ring prompt continuation");
                compare_snapshot(
                    snapshot(&mut e, slot, token),
                    &reference[case][2],
                    &format!("cold slot={slot} continuation"),
                );
                assert_eq!(e.live_ring_stats(), Some(stats));
                let (free, total) = be.mem_info().unwrap();
                eprintln!(
                    "cold slot={slot} prefix={prefix} rings={stats:?} used_bytes={}",
                    total - free
                );
            }
        }
        let retained = e.live_ring_stats();
        for slot in 0..e.batch() {
            e.begin_slot(slot, 1).expect("clear baseline positions");
        }
        assert_eq!(
            e.live_ring_stats(),
            retained,
            "slot reset must retain ring backing"
        );

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
            if effective_live || lazy_rings {
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
        if lazy_rings && live {
            assert_eq!(e.live_ring_stats(), retained);
        }
    }
}

#[test]
fn packed_prefill_chunk_sizes_preserve_full_logits() {
    let _env = common::env_guard();
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        return;
    }
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("PLOW_GPU_ASSETS"));
    assert!(
        !assets.join("block.json").exists(),
        "requires full-model assets"
    );
    let tokenizer = plowrt::text::tokenizer::load_tokenizer(&assets);
    let prompt = tokenizer.encode(&format!(
        "{}The capital of France is",
        "A short sentence. ".repeat(260)
    ));
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(e.pf_batch_enabled() && e.batch() >= 4);
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).unwrap();
    let live = plowrt::memory::vmm::LiveKvLayout::manifest(&blob, &raw)
        .unwrap()
        .unwrap();
    let mut kv_reference: Vec<Vec<u8>> = Vec::new();
    let mut references = Vec::new();
    let mut teacher = vec![*prompt.last().unwrap()];
    let mut output = Vec::new();
    for (case, chunk, slots) in [
        ("large", 1024, vec![0]),
        ("small", 128, vec![0]),
        ("packed", 128, vec![0, 3]),
    ] {
        for slot in 0..e.batch() {
            e.begin_slot(slot, prompt.len() + 10).unwrap();
        }
        let chunk = chunk.min(e.pf_max_rows() / slots.len());
        for start in (0..prompt.len() - 1).step_by(chunk) {
            let len = chunk.min(prompt.len() - 1 - start);
            let reqs: Vec<_> = slots
                .iter()
                .map(|&slot| plowrt::exec::gpu::PfBatchReq {
                    slot,
                    prompt: &prompt,
                    c0: start,
                    len,
                })
                .collect();
            e.prefill_batched(&reqs).unwrap();
        }
        for (ci, cache) in live.caches.iter().enumerate() {
            let name = &blob.tensors[cache.pair[0] as usize].name;
            let mut raw = vec![0; cache.hd as usize * 64 * 2];
            e.read_tensor_range(name, 0, &mut raw).unwrap();
            if case == "large" {
                kv_reference.push(raw);
            } else {
                assert!(
                    raw == kv_reference[ci],
                    "{case}: {name} first 64 KV rows differ"
                );
            }
        }
        for step in 0..9 {
            let feeds: Vec<_> = slots.iter().map(|&slot| (slot, teacher[step])).collect();
            e.step_slots(&feeds, &mut output).unwrap();
            if case == "large" {
                teacher.push(output[0]);
                references.push(snapshot(&mut e, 0, output[0]));
            } else {
                for (ri, &slot) in slots.iter().enumerate() {
                    compare_snapshot(
                        snapshot(&mut e, slot, output[ri]),
                        &references[step],
                        &format!("{case} slot={slot} step={step}"),
                    );
                }
            }
        }
        eprintln!("packed full model: case={case} slots={slots:?}, 9 teacher-forced full-logit rows and {} KV samples exact", live.caches.len());
    }
}

#[test]
#[ignore = "requires free H100 and paired Gemma-4 FP8 role/interpreter assets"]
fn w8a16_m1_role_matches_interpreter_on_real_prompts() {
    const GENERATION_STEPS: usize = 8;
    const MAX_PREFILL_REL_L2: f64 = 1.0e-2;
    let _env = common::env_guard();
    assert_eq!(std::env::var("PLOW_GPU_TEST").as_deref(), Ok("1"));
    let _config = common::EnvScope::set(&[
        ("PLOW_VMM_LIVE", "0"),
        ("PLOW_VMM_PREFIX", "0"),
        ("PLOW_PREFIX_CACHE", "0"),
        ("PLOW_PF_BATCH", "0"),
        ("PLOW_TOKEN_BATCH", "0"),
        ("PLOW_MULTISTEP", "0"),
        ("PLOW_KV_POOL_MIB", "0"),
    ]);
    let baseline = PathBuf::from(std::env::var("TEST_W8A16_M1_BASELINE").unwrap());
    let candidate = PathBuf::from(std::env::var("TEST_W8A16_M1_ASSETS").unwrap());
    let output = PathBuf::from(std::env::var("TEST_W8A16_M1_LOGITS_OUT").unwrap());
    assert!(output.is_absolute() && output.starts_with("/tmp"));

    let load_packet = |dir: &std::path::Path| {
        let path = plowrt::asset::devblob::DevBlob::find_in_dir(dir)
            .unwrap()
            .unwrap();
        let raw = std::fs::read(path).unwrap();
        let blob = plowrt::asset::devblob::DevBlob::parse(&raw).unwrap();
        (blob, raw)
    };
    let (reference_blob, reference_raw) = load_packet(&baseline);
    let (candidate_blob, candidate_raw) = load_packet(&candidate);
    assert!(reference_blob
        .reserved_metadata(&reference_raw, plow_asset::segment_roles::SECTION)
        .unwrap()
        .is_none());
    let role_bytes = candidate_blob
        .reserved_metadata(&candidate_raw, plow_asset::segment_roles::SECTION)
        .unwrap()
        .expect("candidate role metadata");
    let roles = plow_asset::segment_roles::SegmentRoles::from_bytes(role_bytes).unwrap();
    assert_eq!(
        roles.objects.keys().copied().collect::<Vec<_>>(),
        [plow_asset::segment_roles::W8A16_PREFILL_M1]
    );
    let role_segments = roles
        .programs
        .iter()
        .flat_map(|program| &program.roles)
        .filter(|&&role| role == plow_asset::segment_roles::W8A16_PREFILL_M1)
        .count();
    assert!(role_segments > 0);
    assert!(roles.programs.iter().all(|program| {
        candidate_blob.progs[program.index].t == 1
            && program
                .roles
                .contains(&plow_asset::segment_roles::W8A16_PREFILL_M1)
    }));
    assert!(reference_blob
        .tensors
        .iter()
        .map(|tensor| (&tensor.name, tensor.bytes))
        .eq(candidate_blob
            .tensors
            .iter()
            .map(|tensor| (&tensor.name, tensor.bytes))));
    assert_eq!(reference_blob.progs.len(), candidate_blob.progs.len());
    for (reference, actual) in reference_blob.progs.iter().zip(&candidate_blob.progs) {
        assert_eq!(reference.t, actual.t);
        assert_eq!(reference.insts, actual.insts);
        assert_eq!(reference.n_counter, actual.n_counter);
        assert_eq!(reference.stream, actual.stream);
        assert_eq!(reference.stream_ofs, actual.stream_ofs);
        assert_eq!(reference.stream_len, actual.stream_len);
        assert_eq!(reference.waits, actual.waits);
        assert_eq!(reference.succs, actual.succs);
        assert_eq!(reference.gq_stream, actual.gq_stream);
        assert_eq!(reference.gq_seg_ofs, actual.gq_seg_ofs);
    }
    drop((reference_blob, reference_raw, candidate_blob, candidate_raw));

    let tokenizer = plowrt::text::tokenizer::load_tokenizer(&candidate);
    let texts = [
        "Paris",
        "Explain why the sky appears blue during the day in two clear sentences.",
        "A compiler maps a model graph into packets while preserving every dependency and tensor extent. Describe how to test the resulting GPU program for numerical correctness.",
        "Modern language model serving combines matrix multiplication, attention, cache management, and scheduling. A useful benchmark keeps the prompt natural, compares every vocabulary logit, and follows the same greedy trajectory for several decode steps. This catches small numerical changes that a synthetic projection test can miss.",
    ];
    let lengths = [1usize, 17, 65, 257];
    let prompts: Vec<Vec<u32>> = texts
        .iter()
        .zip(lengths)
        .map(|(text, length)| {
            let seed = tokenizer.encode(text);
            assert!(!seed.is_empty());
            seed.iter().copied().cycle().take(length).collect()
        })
        .collect();

    struct Frame {
        token: u32,
        logits: Vec<f32>,
    }
    let run = |dir: &std::path::Path, be: &Arc<CudaBackend>, teacher: Option<&Vec<Vec<Frame>>>| {
        let mut engine = GpuEngine::load(Arc::clone(be), dir, &dir.join("checkpoint")).unwrap();
        assert!(engine.has_prefill());
        let mut trajectories = Vec::with_capacity(prompts.len());
        let mut decoded = Vec::new();
        for (case, prompt) in prompts.iter().enumerate() {
            engine
                .begin_slot(0, prompt.len() + GENERATION_STEPS)
                .unwrap();
            let cap = prompt.len().saturating_sub(1).max(1);
            let mut token = loop {
                match engine.prefill_chunk(0, prompt, cap).unwrap() {
                    plowrt::exec::gpu::PrefillStep::Progress(_) => {}
                    plowrt::exec::gpu::PrefillStep::Done(token) => break token,
                }
            };
            let mut frames = Vec::with_capacity(GENERATION_STEPS);
            for step in 0..GENERATION_STEPS {
                let mut logits = Vec::new();
                engine.logits_row(0, &mut logits).unwrap();
                assert_eq!(logits.len(), 262_144);
                assert!(logits.iter().all(|value| value.is_finite()));
                frames.push(Frame { token, logits });
                if step + 1 != GENERATION_STEPS {
                    let input = teacher.map_or(token, |frames| frames[case][step].token);
                    engine.step_slots(&[(0, input)], &mut decoded).unwrap();
                    token = decoded[0];
                }
            }
            trajectories.push(frames);
            engine.retire_slot(0, false);
        }
        trajectories
    };

    let be = Arc::new(CudaBackend::new(0).unwrap());
    let reference = run(&baseline, &be, None);
    let actual = run(&candidate, &be, Some(&reference));
    let mut records = Vec::new();
    let mut max_rel_l2 = 0.0f64;
    let mut max_first_frame_rel_l2 = 0.0f64;
    let mut rel_l2_sum = 0.0f64;
    let mut greedy_matches = 0usize;
    let mut first_frame_greedy_matches = 0usize;
    for (case, (actual_frames, reference_frames)) in actual.iter().zip(&reference).enumerate() {
        for (step, (actual, reference)) in actual_frames.iter().zip(reference_frames).enumerate() {
            let mut squared_error = 0.0f64;
            let mut squared_reference = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&got, &want) in actual.logits.iter().zip(&reference.logits) {
                let error = f64::from(got) - f64::from(want);
                squared_error += error * error;
                squared_reference += f64::from(want) * f64::from(want);
                max_abs = max_abs.max(error.abs());
            }
            let rel_l2 = (squared_error / squared_reference).sqrt();
            max_rel_l2 = max_rel_l2.max(rel_l2);
            if step == 0 {
                max_first_frame_rel_l2 = max_first_frame_rel_l2.max(rel_l2);
            }
            rel_l2_sum += rel_l2;
            let greedy_equal = actual.token == reference.token;
            greedy_matches += usize::from(greedy_equal);
            if step == 0 {
                first_frame_greedy_matches += usize::from(greedy_equal);
            }
            records.push(serde_json::json!({
                "case": case,
                "prompt_tokens": prompts[case].len(),
                "step": step,
                "rel_l2": rel_l2,
                "max_abs": max_abs,
                "candidate_token": actual.token,
                "reference_token": reference.token,
                "greedy_equal": greedy_equal,
            }));
        }
    }
    let frames = records.len();
    let report = serde_json::json!({
        "model": "google/gemma-4-12B-it",
        "precision": "W8A16 FP8 weights / BF16 activations",
        "candidate_packet_sha256": plow_asset::decode_objects::image_sha256(&std::fs::read(candidate.join("model.pkt")).unwrap()),
        "reference_packet_sha256": plow_asset::decode_objects::image_sha256(&std::fs::read(baseline.join("model.pkt")).unwrap()),
        "role_segments": role_segments,
        "prompt_lengths": lengths,
        "frames": frames,
        "greedy_matches": greedy_matches,
        "first_frame_greedy_matches": first_frame_greedy_matches,
        "max_rel_l2": max_rel_l2,
        "max_first_frame_rel_l2": max_first_frame_rel_l2,
        "mean_rel_l2": rel_l2_sum / frames as f64,
        "criteria": {
            "max_first_frame_rel_l2": MAX_PREFILL_REL_L2,
            "required_first_frame_greedy_matches": prompts.len(),
            "required_teacher_forced_greedy_matches": frames,
        },
        "snapshots": records,
    });
    std::fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    assert!(
        first_frame_greedy_matches == prompts.len()
            && max_first_frame_rel_l2 <= MAX_PREFILL_REL_L2
            && greedy_matches == frames,
        "qualification failed: prefill greedy {first_frame_greedy_matches}/{}, teacher-forced greedy {greedy_matches}/{frames}, max prefill rel-L2={max_first_frame_rel_l2:e} (limit {MAX_PREFILL_REL_L2:e})",
        prompts.len(),
    );
    eprintln!(
        "W8A16 M1 role: {role_segments} segments, {greedy_matches}/{frames} teacher-forced greedy frames, max prefill rel-L2={max_first_frame_rel_l2:e}, max propagated rel-L2={max_rel_l2:e}, mean rel-L2={:e}",
        rel_l2_sum / frames as f64
    );
}
