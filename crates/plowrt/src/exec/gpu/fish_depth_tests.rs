use super::*;
use crate::asset::devblob::DevBlob;
use crate::device::cuda::CudaBackend;
use plow_asset::token_batch::{Phase, Request, Selection};
use std::sync::Arc;

fn upload_inputs(
    e: &mut GpuEngine,
    m: &Manifest,
    plan: &plow_asset::token_batch::Plan,
) -> Result<()> {
    let k = m.codebooks as usize;
    let mut codes = vec![0u32; e.pf_max_rows() * k];
    let mut index = vec![u32::MAX - 1; e.pf_max_rows()];
    // Every request uses a distinct, repeatable code row and an overlay at prompt position 1.
    let hidden = e.packed_terminal.as_ref().unwrap().hidden_rows().1 as usize / 2;
    let mut overlays = Vec::new();
    for (span, phase) in plan.spans.iter().zip(&plan.phases) {
        if *phase == Phase::Prefill {
            for r in 0..span.n_rows as usize {
                let at = (span.row0 as usize + r) * k;
                for c in 0..k {
                    codes[at + c] = (span.slot + span.kv_row0 + r as u32 + c as u32) % 16;
                }
                if span.kv_row0 + r as u32 == 1 && e.handle_of("in.encoder_overlay").is_some() {
                    index[span.row0 as usize + r] = 0;
                    if overlays.is_empty() {
                        overlays.extend((0..hidden).map(|j| ((j % 17) as f32 - 8.0) * 0.001));
                    }
                }
            }
        }
    }
    e.write_tensor("in.frame_codes", 0, bytemuck::cast_slice(&codes))?;
    if e.handle_of("in.encoder_overlay").is_some() {
        if !overlays.is_empty() {
            e.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(&overlays))?;
        }
        e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&index))?;
    }
    e.stage_depth_decode_codes(plan)
}

fn draw(e: &mut GpuEngine, outputs: &[(u32, u32)], step: u32) {
    let rows: Vec<_> = outputs
        .iter()
        .map(|&(slot, _)| FishRow {
            slot: slot as usize,
            hidden_row: slot,
            seed: 193 + u64::from(slot),
            step: if slot == 3 { step } else { 0 },
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            ras_temperature: 1.0,
            ras_top_p: 1.0,
            repetition_penalty: 1.0,
            cfg: None,
        })
        .collect();
    let owners: Vec<_> = outputs.iter().map(|o| o.0).collect();
    let mut codes = vec![u32::MAX];
    e.fish_depth_step_compact(&rows, &owners, &mut codes)
        .unwrap();
    let m = &e.fish.as_ref().unwrap().manifest;
    assert_eq!(
        codes[0],
        u32::MAX,
        "depth output must append to the caller's buffer"
    );
    assert_eq!(codes.len(), 1 + rows.len() * m.codebooks as usize);
    assert!(codes[1..]
        .chunks_exact(m.codebooks as usize)
        .all(|c| c[0] <= m.stop_class));
}

#[test]
#[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, PLOW_VMM_PREFIX=1, RUST_MIN_STACK=67108864; run through gpuq"]
fn speech_mixed_batch_maps_frames_and_reuses_only_matching_content() {
    let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(e.token_batch_enabled() && e.packed_depth_enabled());
    let shared = e.vmm_prefix_enabled();
    assert!(shared || e.slot_prefix_reuse_supported());
    let live = e.live_slot_prefix_supported();
    if crate::config::RuntimeConfig::get().nv.vmm_live {
        assert!(
            live,
            "explicit live KV must exercise private live prefix reuse"
        );
    }
    eprintln!(
        "cache mode: shared VMM={shared}, private live={live}, slot prefix={}",
        e.slot_prefix_reuse_supported()
    );
    assert!(e.batch >= 4);
    let m = e.fish.as_ref().unwrap().manifest.clone();
    let contract = m.packed.as_ref().unwrap();
    let (src, pitch) = e.packed_terminal.as_ref().unwrap().hidden_rows();

    // Exact device gather oracle, including a permutation that corrupts an in-place scatter.
    let owners = [3, 0, 2];
    let hidden: Vec<u16> = (0..3 * pitch as usize / 2)
        .map(|i| (i % 8192) as u16)
        .collect();
    let logits: Vec<u16> = (0..3 * e.vocab).map(|i| ((i + 97) % 8192) as u16).collect();
    e.be.memcpy_htod(src, bytemuck::cast_slice(&hidden))
        .unwrap();
    e.be.memcpy_htod(e.devp[e.t_logits].base, bytemuck::cast_slice(&logits))
        .unwrap();
    let mut d = e.fish.take().unwrap();
    d.compact.as_mut().unwrap().launch(&e, &owners).unwrap();
    e.be.stream_synchronize(&e.stream).unwrap();
    let mut actual = vec![0u16; pitch as usize / 2];
    let mut actual_logits = vec![0u16; e.vocab];
    for (r, &slot) in owners.iter().enumerate() {
        e.be.download(
            &e.devp[contract.hidden as usize],
            u64::from(slot) * pitch,
            bytemuck::cast_slice_mut(&mut actual),
        )
        .unwrap();
        assert_eq!(actual, hidden[r * actual.len()..(r + 1) * actual.len()]);
        if let Some(scratch) = &d.compact.as_ref().unwrap().logits {
            e.be.download(
                scratch,
                u64::from(slot) * e.vocab as u64 * 2,
                bytemuck::cast_slice_mut(&mut actual_logits),
            )
            .unwrap();
            assert_eq!(actual_logits, logits[r * e.vocab..(r + 1) * e.vocab]);
        }
    }
    let mut original = vec![0u16; logits.len()];
    e.be.download(
        &e.devp[e.t_logits],
        0,
        bytemuck::cast_slice_mut(&mut original),
    )
    .unwrap();
    assert_eq!(original, logits);
    e.fish = Some(d);

    let mut a: Vec<_> = (0..41)
        .map(|i| (31 + i * 7919) % contract.input_vocab)
        .collect();
    let mut b: Vec<_> = (0..37)
        .map(|i| (97 + i * 7919) % contract.input_vocab)
        .collect();
    if let Some(token) = contract.decode_token {
        a[3] = token;
        b[3] = token;
    }
    let keys_a: Vec<u64> = (0..a.len()).map(|i| i as u64 + 900).collect();
    let keys_b: Vec<u64> = (0..b.len()).map(|i| i as u64 + 1900).collect();
    for (slot, prompt, keys) in [(3, &a, &keys_a), (0, &b, &keys_b)] {
        if shared {
            assert!(e.stage_speech_prefix(slot, keys));
        }
        assert_eq!(
            e.admit_packed_slot(slot, prompt, prompt.len() + 8).unwrap(),
            Some(0)
        );
        e.fish_reset_slot(slot).unwrap();
        if m.sampler == Sampler::Maya2 {
            let text =
                std::fs::read_to_string(assets.join("checkpoint/maya2_controller.json")).unwrap();
            let ctl = crate::tts::maya2::Controller::parse(&text).unwrap();
            let plan = ctl.plan(
                "A short correctness check.",
                "en",
                Some(8),
                ctl.cb0,
                193 + slot as u64,
            );
            e.maya2_bind(slot, plan, None).unwrap();
        }
    }
    let requests: Vec<_> = [(3, a.as_slice()), (0, b.as_slice())]
        .into_iter()
        .map(|(slot, tokens)| Request {
            id: slot,
            slot,
            state_slot: slot,
            generation: e.slot_generation(slot as usize).unwrap(),
            phase: Phase::Prefill,
            tokens,
            prompt_len: tokens.len() as u32,
            selection: Selection::default(),
        })
        .collect();
    let mut outputs = Vec::new();
    e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
        upload_inputs(e, &m, p)
    })
    .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!((e.pos[3], e.pos[0]), (41, 37));
    let mut fresh_prompt_logits = Vec::new();
    let row = outputs.iter().position(|o| o.0 == 0).unwrap();
    e.logits_row(row, &mut fresh_prompt_logits).unwrap();
    if !shared {
        let epoch = e.slot_prefix_epoch();
        e.keep_slot_prefix(3, epoch, keys_a.clone());
        e.keep_slot_prefix(0, epoch, keys_b.clone());
    }
    draw(&mut e, &outputs, 0);

    // A continuing decode row rides beside a new prompt, through the hd128 split-attention arm.
    for step in 1..=2 {
        e.retire_slot(0, false);
        if shared {
            assert!(e.stage_speech_prefix(0, &keys_b));
        } else {
            let cached = e.reuse_slot_prefix(0, &keys_b);
            assert_eq!(cached, b.len() - 1);
            assert_eq!(e.resume_slot(0, cached), cached);
        }
        let frontier = e.admit_packed_slot(0, &b, b.len() + 8).unwrap().unwrap();
        assert!(
            frontier >= 32 && frontier < b.len(),
            "matching VMM content did not attach"
        );
        e.fish_reset_slot(0).unwrap();
        if m.sampler == Sampler::Maya2 {
            let text =
                std::fs::read_to_string(assets.join("checkpoint/maya2_controller.json")).unwrap();
            let ctl = crate::tts::maya2::Controller::parse(&text).unwrap();
            e.maya2_bind(
                0,
                ctl.plan("A short correctness check.", "en", Some(8), ctl.cb0, 193),
                None,
            )
            .unwrap();
        }
        let input = [contract.input_vocab - 3];
        let mut reference = Vec::new();
        let mut tokens = Vec::new();
        e.step_slots(&[(3, input[0])], &mut tokens).unwrap();
        e.logits_row(3, &mut reference).unwrap();
        let mut reference_hidden = vec![0u16; pitch as usize / 2];
        e.be.download(
            &e.devp[contract.hidden as usize],
            3 * pitch,
            bytemuck::cast_slice_mut(&mut reference_hidden),
        )
        .unwrap();
        e.rewind_slot(3, a.len() + step as usize - 1).unwrap();
        let requests = [
            Request {
                id: 0,
                slot: 0,
                state_slot: 0,
                generation: e.slot_generation(0).unwrap(),
                phase: Phase::Prefill,
                tokens: &b[frontier..],
                prompt_len: b.len() as u32,
                selection: Selection::default(),
            },
            Request {
                id: 3,
                slot: 3,
                state_slot: 3,
                generation: e.slot_generation(3).unwrap(),
                phase: Phase::Decode,
                tokens: &input,
                prompt_len: a.len() as u32,
                selection: Selection::default(),
            },
        ];
        e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
            upload_inputs(e, &m, p)
        })
        .unwrap();
        assert_eq!(
            outputs[0].0, 3,
            "the planner must put decode before prefill"
        );
        assert_eq!((e.pos[3], e.pos[0]), (41 + step, 37));
        if !shared {
            e.keep_slot_prefix(0, e.slot_prefix_epoch(), keys_b.clone());
        }
        let mut actual = Vec::new();
        e.logits_row(0, &mut actual).unwrap();
        let mut actual_hidden = vec![0u16; pitch as usize / 2];
        e.be.memcpy_dtoh(bytemuck::cast_slice_mut(&mut actual_hidden), src)
            .unwrap();
        let (mut dot, mut rr, mut aa, mut dd) = (0f64, 0f64, 0f64, 0f64);
        for (&r, &a) in reference_hidden.iter().zip(&actual_hidden) {
            let r = f64::from(f32::from_bits(u32::from(r) << 16));
            let a = f64::from(f32::from_bits(u32::from(a) << 16));
            dot += r * a;
            rr += r * r;
            aa += a * a;
            dd += (r - a) * (r - a);
        }
        let max_error = actual
            .iter()
            .zip(&reference)
            .map(|(a, r)| (a - r).abs())
            .fold(0f32, f32::max);
        let top = |l: &[f32]| {
            let mut ids: Vec<_> = l.iter().copied().enumerate().collect();
            ids.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            ids[..4].to_vec()
        };
        eprintln!("hidden cosine={} relative RMS={}, logits max_error={max_error}, reference top={:?}, actual top={:?}",
            dot / (rr * aa).sqrt(), (dd / rr).sqrt(), top(&reference), top(&actual));
        assert!(
            dot / (rr * aa).sqrt() > 0.999,
            "mixed decode hidden state differs"
        );
        assert!(actual.iter().chain(&reference).all(|v| v.is_finite()));
        let softmax = |values: &[f32]| {
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f64> = values.iter().map(|&v| f64::from(v - max).exp()).collect();
            let sum = p.iter().sum::<f64>();
            p.iter_mut().for_each(|p| *p /= sum);
            p
        };
        let (p, q) = (softmax(&reference), softmax(&actual));
        let kl: f64 = p
            .iter()
            .zip(q)
            .filter(|(p, _)| **p > 0.0)
            .map(|(&p, q)| p * (p / q).ln())
            .sum();
        eprintln!("mixed decode vs ordinary decode full-logit KL = {kl}");
        assert!(kl < 1e-3, "mixed decode differs from ordinary decode");
        let mut cached_prompt_logits = Vec::new();
        let row = outputs.iter().position(|o| o.0 == 0).unwrap();
        e.logits_row(row, &mut cached_prompt_logits).unwrap();
        let (p, q) = (
            softmax(&fresh_prompt_logits),
            softmax(&cached_prompt_logits),
        );
        let cached_kl: f64 = p
            .iter()
            .zip(q)
            .filter(|(p, _)| **p > 0.0)
            .map(|(&p, q)| p * (p / q).ln())
            .sum();
        eprintln!("retained prompt vs fresh full-logit KL = {cached_kl}");
        assert!(
            cached_kl < 1e-3,
            "retained prefix changed the prompt output"
        );
        assert_eq!(
            reference[crate::text::sample::argmax(&reference) as usize],
            reference[crate::text::sample::argmax(&actual) as usize],
            "mixed decode must choose a reference maximum, including BF16 ties"
        );
        draw(&mut e, &outputs, step);
    }

    // A retained tail near max_ctx must also fit the packet's unmasked prefill bucket.
    if !shared {
        let n = e.max_ctx - 8;
        let mut prompt: Vec<_> = (0..n)
            .map(|i| (31 + i as u32 * 7919) % contract.input_vocab)
            .collect();
        if let Some(token) = contract.decode_token {
            prompt[3] = token;
        }
        let keys: Vec<_> = (0..n as u64).map(|i| i + 9000).collect();
        e.admit_packed_slot(2, &prompt, n + 8).unwrap().unwrap();
        while (e.pos[2] as usize) < n {
            let start = e.pos[2] as usize;
            let end = (start + e.pf_max_rows()).min(n);
            let requests = [Request {
                id: 2,
                slot: 2,
                state_slot: 2,
                generation: e.slot_generation(2).unwrap(),
                phase: Phase::Prefill,
                tokens: &prompt[start..end],
                prompt_len: n as u32,
                selection: Selection::default(),
            }];
            e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
                upload_inputs(e, &m, p)
            })
            .unwrap();
        }
        let mut fresh = Vec::new();
        e.logits_row(0, &mut fresh).unwrap();
        e.keep_slot_prefix(2, e.slot_prefix_epoch(), keys.clone());
        e.retire_slot(2, false);
        let cached = e.reuse_slot_prefix(2, &keys);
        assert!(cached > 0 && cached < n);
        assert_eq!(e.resume_slot(2, cached), cached);
        assert_eq!(
            e.admit_packed_slot(2, &prompt, n + 8).unwrap(),
            Some(cached)
        );
        let requests = [Request {
            id: 2,
            slot: 2,
            state_slot: 2,
            generation: e.slot_generation(2).unwrap(),
            phase: Phase::Prefill,
            tokens: &prompt[cached..],
            prompt_len: n as u32,
            selection: Selection::default(),
        }];
        e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
            upload_inputs(e, &m, p)
        })
        .unwrap();
        let mut resumed = Vec::new();
        e.logits_row(0, &mut resumed).unwrap();
        assert_eq!(
            fresh, resumed,
            "near-limit retained tail changed the output"
        );
    }

    e.retire_slot(0, false);
    let mut changed = keys_b.clone();
    changed[0] ^= 1;
    if shared {
        assert!(e.stage_speech_prefix(0, &changed));
    } else {
        assert_eq!(e.reuse_slot_prefix(0, &changed), 0);
    }
    assert_eq!(e.admit_packed_slot(0, &b, b.len() + 8).unwrap(), Some(0));
    e.retire_slot(0, false);
    e.begin_slot(0, b.len() + 8).unwrap();
    e.set_prefix_attach(0, true);
    assert_eq!(
        e.attach_prompt(0, &b).unwrap(),
        0,
        "raw tokens aliased speech identities"
    );
    let input = [contract.input_vocab - 3];
    let requests = [Request {
        id: 3,
        slot: 3,
        state_slot: 3,
        generation: e.slot_generation(3).unwrap(),
        phase: Phase::Decode,
        tokens: &input,
        prompt_len: a.len() as u32,
        selection: Selection::default(),
    }];
    let before = e.pos.clone();
    assert!(
        e.token_batch_step(&requests, &mut outputs).is_err(),
        "unstaged speech was accepted"
    );
    assert_eq!(e.pos, before);
}

fn logit_kl(reference: &[f32], actual: &[f32]) -> f64 {
    assert_eq!(reference.len(), actual.len());
    assert!(reference.iter().chain(actual).all(|v| v.is_finite()));
    let softmax = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f64> = values.iter().map(|&v| f64::from(v - max).exp()).collect();
        let sum = p.iter().sum::<f64>();
        p.iter_mut().for_each(|p| *p /= sum);
        p
    };
    softmax(reference)
        .into_iter()
        .zip(softmax(actual))
        .filter(|(p, _)| *p > 0.0)
        .map(|(p, q)| p * (p / q).ln())
        .sum()
}

#[derive(serde::Deserialize)]
struct SpeechTeacherCase {
    id: String,
    text: String,
    ids: Vec<u32>,
    voice: String,
    language: String,
    codes: Vec<Vec<u32>>,
}

#[derive(serde::Deserialize)]
struct SpeechTeacherInputs {
    version: u32,
    frames: usize,
    dual_stream: plow_asset::dual_stream::DualStream,
    cases: Vec<SpeechTeacherCase>,
}

fn teacher_codes(
    e: &mut GpuEngine,
    layouts: &[plow_asset::dual_stream::Layout],
    cases: &[&SpeechTeacherCase],
    k: usize,
    plan: &plow_asset::token_batch::Plan,
) -> Result<()> {
    let mut codes = vec![0u32; e.pf_max_rows() * k];
    for span in &plan.spans {
        let slot = span.slot as usize;
        let layout = &layouts[slot];
        for row in 0..span.n_rows as usize {
            let pos = span.kv_row0 as usize + row;
            let source = if pos < layout.ids.len() {
                &layout.codes[pos * k..(pos + 1) * k]
            } else {
                &cases[slot].codes[pos - layout.ids.len()]
            };
            let start = (span.row0 as usize + row) * k;
            codes[start..start + k].copy_from_slice(source);
        }
    }
    e.write_tensor("in.frame_codes", 0, bytemuck::cast_slice(&codes))?;
    e.stage_depth_decode_codes(plan)
}

#[test]
#[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, SPEECH_BATCH_TEST_CASES, RUST_MIN_STACK=67108864; run through gpuq"]
fn speech_teacher_forced_backbone_preserves_logits_at_serving_concurrencies() {
    let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
    let path = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_CASES").unwrap());
    let inputs: SpeechTeacherInputs =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(inputs.version, 1);
    assert!(inputs.cases.len() >= 16 && inputs.frames >= 16);
    let dual = plow_asset::dual_stream::DualStream::parse(
        &std::fs::read_to_string(assets.join("checkpoint/dual_stream.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(dual, inputs.dual_stream);
    let mut e = GpuEngine::load(
        Arc::new(CudaBackend::new(0).unwrap()),
        &assets,
        &assets.join("checkpoint"),
    )
    .unwrap();
    assert!(e.token_batch_enabled() && e.packed_depth_enabled() && e.batch >= 32);
    let m = e.fish.as_ref().unwrap().manifest.clone();
    assert!(m.packed.as_ref().unwrap().decode_token.is_none());
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let pipelines: plow_asset::packet_pipeline::PacketPipelines = serde_json::from_slice(
        blob.reserved_metadata(&raw, plow_asset::packet_pipeline::SECTION)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let speech = crate::tts::SpeechContract::from_pipeline(
        pipelines
            .pipelines
            .iter()
            .find(|p| p.driver == crate::tts::DRIVER)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(speech.dual_stream.as_ref(), Some(&dual));
    let sample_row = |slot: usize, hidden_row: u32| FishRow {
        slot,
        hidden_row,
        seed: 1,
        step: 0,
        temperature: speech.temperature,
        top_p: speech.top_p,
        top_k: speech.top_k as u32,
        ras_temperature: 1.0,
        ras_top_p: 1.0,
        repetition_penalty: speech.repetition_penalty,
        cfg: None,
    };
    let k = m.codebooks as usize;
    assert_eq!(k, dual.frame_codes);
    for case in &inputs.cases {
        assert!(case.codes.len() >= inputs.frames);
        assert!(case
            .codes
            .iter()
            .all(|c| { c.len() == k && c.iter().all(|&v| v < m.semantic_tokens) }));
    }
    if e.handle_of("in.encoder_overlay_index").is_some() {
        let keep = vec![u32::MAX - 1; e.pf_max_rows()];
        e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&keep))
            .unwrap();
    }
    let mut groups = Vec::new();
    let mut failures = Vec::new();
    for n in [1, 8, 32] {
        for offset in (0..inputs.cases.len()).step_by(n) {
            let cases: Vec<_> = (0..n)
                .map(|slot| &inputs.cases[(offset + slot) % inputs.cases.len()])
                .collect();
            let layouts: Vec<_> = cases
                .iter()
                .map(|c| {
                    dual.layout(&c.ids, &[], &c.voice, &c.language, true)
                        .unwrap()
                })
                .collect();
            let admit = |e: &mut GpuEngine| {
                for (slot, layout) in layouts.iter().enumerate() {
                    assert_eq!(
                        e.admit_packed_slot(
                            slot,
                            &layout.ids,
                            layout.ids.len() + inputs.frames + 1
                        )
                        .unwrap(),
                        Some(0)
                    );
                    e.set_prefix_attach(slot, false);
                }
            };
            let feed_ids = |step: usize| -> Vec<[u32; 1]> {
                layouts
                    .iter()
                    .map(|l| [l.feed.get(step).copied().unwrap_or(l.tail)])
                    .collect()
            };
            let seed = |e: &mut GpuEngine, step: usize| {
                let seeds: Vec<_> = cases
                    .iter()
                    .enumerate()
                    .map(|(slot, case)| (slot, case.codes[step].as_slice()))
                    .collect();
                e.fish_seed_slots(&seeds).unwrap();
            };
            let read_native = |e: &mut GpuEngine| -> Vec<Vec<f32>> {
                (0..n)
                    .map(|slot| {
                        let mut row = Vec::new();
                        e.logits_row(slot, &mut row).unwrap();
                        row
                    })
                    .collect()
            };
            let mut compare =
                |arm: &str, step: usize, reference: &[Vec<f32>], actual: &[Vec<f32>]| {
                    for slot in 0..n {
                        let kl = logit_kl(&reference[slot], &actual[slot]);
                        if kl >= 1e-3 {
                            eprintln!(
                                "teacher c{n} offset{offset} {} {arm} step{step}: KL={kl}",
                                cases[slot].id
                            );
                            failures.push(serde_json::json!({"concurrency":n,"offset":offset,
                            "case":cases[slot].id,"slot":slot,"arm":arm,"step":step,"kl":kl}));
                        }
                    }
                };
            admit(&mut e);
            let mut first = vec![Vec::new(); n];
            let mut cold_codes = vec![Vec::new(); n];
            for (slot, layout) in layouts.iter().enumerate() {
                e.write_tensor("in.frame_codes", 0, bytemuck::cast_slice(&layout.codes))
                    .unwrap();
                e.prefill_slot(slot, &layout.ids).unwrap();
                e.logits_row(0, &mut first[slot]).unwrap();
                e.fish_reset_slot(slot).unwrap();
                e.fish_logits_to_slot(slot).unwrap();
                e.fish_depth_step(
                    &[sample_row(slot, e.prefill_terminal_row())],
                    &mut cold_codes[slot],
                )
                .unwrap();
                assert_eq!(cold_codes[slot].len(), k);
            }
            let mut ordinary = vec![first];
            for step in 0..inputs.frames {
                seed(&mut e, step);
                let ids = feed_ids(step);
                let feeds: Vec<_> = (0..n).rev().map(|s| (s, ids[s][0])).collect();
                e.step_slots(&feeds, &mut Vec::new()).unwrap();
                ordinary.push(read_native(&mut e));
            }
            for (slot, layout) in layouts.iter().enumerate() {
                e.rewind_slot(slot, layout.ids.len() - 1).unwrap();
            }
            let seeds: Vec<_> = layouts
                .iter()
                .enumerate()
                .map(|(s, l)| (s, &l.codes[l.codes.len() - k..]))
                .collect();
            e.fish_seed_slots(&seeds).unwrap();
            let feeds: Vec<_> = (0..n)
                .rev()
                .map(|s| (s, *layouts[s].ids.last().unwrap()))
                .collect();
            e.step_slots(&feeds, &mut Vec::new()).unwrap();
            let warm = read_native(&mut e);
            compare("warm-tail", 0, &ordinary[0], &warm);
            let rows: Vec<_> = (0..n).map(|s| sample_row(s, s as u32)).collect();
            let mut warm_codes = Vec::new();
            e.fish_depth_step(&rows, &mut warm_codes).unwrap();
            assert_eq!(warm_codes.len(), n * k);
            for slot in 0..n {
                e.retire_slot(slot, false);
            }
            admit(&mut e);
            let requests: Vec<_> = (0..n)
                .rev()
                .map(|s| Request {
                    id: s as u32,
                    slot: s as u32,
                    state_slot: s as u32,
                    generation: e.slot_generation(s).unwrap(),
                    phase: Phase::Prefill,
                    tokens: &layouts[s].ids,
                    prompt_len: layouts[s].ids.len() as u32,
                    selection: Selection::default(),
                })
                .collect();
            let mut outputs = Vec::new();
            e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
                teacher_codes(e, &layouts, &cases, k, p)
            })
            .unwrap();
            archive_activations(&e, n, &format!("teacher-o{offset}-prefill"), None);
            let read_compact = |e: &mut GpuEngine, outputs: &[(u32, u32)]| -> Vec<Vec<f32>> {
                assert_eq!(outputs.len(), n);
                let mut rows = vec![Vec::new(); n];
                let mut seen = vec![false; n];
                for (row, &(slot, _)) in outputs.iter().enumerate() {
                    assert!(!std::mem::replace(&mut seen[slot as usize], true));
                    e.logits_row(row, &mut rows[slot as usize]).unwrap();
                }
                assert!(seen.iter().all(|&v| v));
                rows
            };
            let mut packed = vec![read_compact(&mut e, &outputs)];
            compare("packed-prefill", 0, &ordinary[0], &packed[0]);
            for slot in 0..n {
                e.fish_reset_slot(slot).unwrap();
            }
            let owners: Vec<_> = outputs.iter().map(|o| o.0).collect();
            let rows: Vec<_> = owners.iter().map(|&s| sample_row(s as usize, s)).collect();
            let mut sampled = Vec::new();
            e.fish_depth_step_compact(&rows, &owners, &mut sampled)
                .unwrap();
            assert_eq!(sampled.len(), n * k);
            let mut packed_codes = vec![Vec::new(); n];
            for (&slot, codes) in owners.iter().zip(sampled.chunks_exact(k)) {
                packed_codes[slot as usize] = codes.to_vec();
            }
            let mut conditional = Vec::new();
            for step in 0..inputs.frames {
                seed(&mut e, step);
                let ids = feed_ids(step);
                let feeds: Vec<_> = (0..n).rev().map(|s| (s, ids[s][0])).collect();
                e.step_slots(&feeds, &mut Vec::new()).unwrap();
                let reference = read_native(&mut e);
                let actual = if n == 1 {
                    reference.clone()
                } else {
                    for (s, layout) in layouts.iter().enumerate() {
                        e.rewind_slot(s, layout.ids.len() + step).unwrap();
                    }
                    let requests: Vec<_> = (0..n)
                        .rev()
                        .map(|s| Request {
                            id: s as u32,
                            slot: s as u32,
                            state_slot: s as u32,
                            generation: e.slot_generation(s).unwrap(),
                            phase: if s + 1 == n {
                                Phase::Prefill
                            } else {
                                Phase::Decode
                            },
                            tokens: &ids[s],
                            prompt_len: (layouts[s].ids.len()
                                + if s + 1 == n { step + 1 } else { 0 })
                                as u32,
                            selection: Selection::default(),
                        })
                        .collect();
                    e.token_batch_step_staged(
                        &requests,
                        &mut outputs,
                        None,
                        &mut |_| {},
                        &mut |e, p| teacher_codes(e, &layouts, &cases, k, p),
                    )
                    .unwrap();
                    archive_activations(&e, n, &format!("teacher-o{offset}-f{step}"), None);
                    read_compact(&mut e, &outputs)
                };
                compare("same-prefix-native", step + 1, &reference, &actual);
                compare(
                    "ordinary-trajectory",
                    step + 1,
                    &ordinary[step + 1],
                    &actual,
                );
                conditional.push(reference);
                packed.push(actual);
            }
            groups.push(serde_json::json!({"concurrency":n,"offset":offset,
                "case_ids": cases.iter().map(|c| &c.id).collect::<Vec<_>>(),
                "texts":cases.iter().map(|c| &c.text).collect::<Vec<_>>(),
                "layouts": layouts.iter().map(|l| serde_json::json!({"ids":l.ids,"codes":l.codes,"feed":l.feed,"tail":l.tail})).collect::<Vec<_>>(),
                "ordinary":ordinary,"warm":warm,"packed":packed,"conditional_native":conditional,
                "cold_first_codes":cold_codes,"warm_first_codes":warm_codes.chunks_exact(k).collect::<Vec<_>>(),
                "packed_first_codes":packed_codes}));
            for slot in 0..n {
                e.retire_slot(slot, false);
            }
        }
    }
    serde_json::to_writer(
        std::fs::File::create(path.with_extension("logits.json")).unwrap(),
        &serde_json::json!({"version":1,"frames":inputs.frames,"groups":groups,"failures":failures}),
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "{} teacher-forced KL outliers; full records in {}",
        failures.len(),
        path.with_extension("logits.json").display()
    );
}

fn kv_vectors(
    e: &GpuEngine,
    blob: &DevBlob,
    positions: &[(usize, usize)],
) -> Vec<(String, Vec<u16>)> {
    let layout = crate::exec::gpu::packet_kv_layout(blob);
    assert!(matches!(
        layout,
        plow_asset::live_kv::LAYOUT_HEAD_MAJOR | plow_asset::live_kv::LAYOUT_ROW_INTERLEAVED
    ));
    let mut tensors = Vec::new();
    for instruction in e
        .h_inst
        .iter()
        .filter(|i| i.op == DevOp::FlashDecode as u16)
    {
        let (heads, stride, width) = (
            instruction.i[2] as usize,
            instruction.i[3] as usize,
            instruction.i[6] as usize,
        );
        for &handle in &instruction.t[3..5] {
            let name = &blob.tensors[handle as usize].name;
            assert!(
                e.handle_of(&format!("{name}_scale")).is_none(),
                "KV capture requires BF16 caches"
            );
            let mut values = vec![0u16; positions.len() * heads * width];
            for (request, &(slot, position)) in positions.iter().enumerate() {
                let row = (position as u32 & instruction.i[7]) as usize;
                assert!(row < stride);
                for head in 0..heads {
                    let offset = if layout == plow_asset::live_kv::LAYOUT_HEAD_MAJOR {
                        ((slot * heads + head) * stride + row) * width
                    } else {
                        ((slot * stride + row) * heads + head) * width
                    };
                    let begin = (request * heads + head) * width;
                    e.read_tensor_range(
                        name,
                        offset as u64 * 2,
                        bytemuck::cast_slice_mut(&mut values[begin..begin + width]),
                    )
                    .unwrap();
                }
            }
            tensors.push((name.clone(), values));
        }
    }
    assert!(!tensors.is_empty());
    tensors
}

fn kv_difference(reference: &[u16], actual: &[u16]) -> (f64, f64) {
    assert_eq!(reference.len(), actual.len());
    let (mut error, mut norm, mut maximum) = (0.0, 0.0, 0f64);
    for (&a, &b) in reference.iter().zip(actual) {
        let (a, b) = (
            f64::from(f32::from_bits(u32::from(a) << 16)),
            f64::from(f32::from_bits(u32::from(b) << 16)),
        );
        assert!(a.is_finite() && b.is_finite());
        error += (a - b) * (a - b);
        norm += a * a;
        maximum = maximum.max((a - b).abs());
    }
    ((error / norm.max(1e-30)).sqrt(), maximum)
}

fn archive_activations(e: &GpuEngine, n: usize, stage: &str, slot: Option<usize>) {
    let Some(directory) = crate::config::RuntimeConfig::get().nv.nv_dump_kv.as_ref() else {
        return;
    };
    if let Some(slot) = slot {
        e.dump_slot_kv(slot, &format!("c{n}-{stage}"), e.pos[slot]);
    }
    let destination = directory.join(format!("c{n}-{stage}"));
    std::fs::create_dir_all(&destination).unwrap();
    for entry in std::fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().starts_with("act-p")
            && entry.file_type().unwrap().is_file()
        {
            std::fs::rename(entry.path(), destination.join(entry.file_name())).unwrap();
        }
    }
}

#[test]
#[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS, RUST_MIN_STACK=67108864; run through gpuq"]
fn speech_mixed_batch_preserves_logits_at_serving_concurrencies() {
    let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
    let blob = DevBlob::parse(&std::fs::read(assets.join("model.pkt")).unwrap()).unwrap();
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(e.token_batch_enabled() && e.packed_depth_enabled() && e.batch >= 32);
    let m = e.fish.as_ref().unwrap().manifest.clone();
    let contract = m.packed.as_ref().unwrap();
    let controller = (m.sampler == Sampler::Maya2).then(|| {
        let text =
            std::fs::read_to_string(assets.join("checkpoint/maya2_controller.json")).unwrap();
        crate::tts::maya2::Controller::parse(&text).unwrap()
    });
    let bind = |e: &mut GpuEngine, slot: usize| {
        e.fish_reset_slot(slot).unwrap();
        if let Some(ctl) = &controller {
            let plan = ctl.plan(
                "A short correctness check.",
                "en",
                Some(8),
                ctl.cb0,
                193 + slot as u64,
            );
            e.maya2_bind(slot, plan, None).unwrap();
        }
    };
    let mut discrepancies = Vec::new();
    for n in [1, 8, 32] {
        let prompts: Vec<Vec<u32>> = (0..n)
            .map(|slot| {
                let mut ids: Vec<_> = (0..7 + slot % 3)
                    .map(|i| (31 + slot as u32 * 101 + i as u32 * 7919) % contract.input_vocab)
                    .collect();
                if let Some(token) = contract.decode_token {
                    ids[3] = token;
                }
                ids
            })
            .collect();
        for (slot, prompt) in prompts.iter().enumerate() {
            e.admit_packed_slot(slot, prompt, prompt.len() + 8)
                .unwrap()
                .unwrap();
            bind(&mut e, slot);
        }
        let requests: Vec<_> = (0..n)
            .rev()
            .map(|slot| Request {
                id: slot as u32,
                slot: slot as u32,
                state_slot: slot as u32,
                generation: e.slot_generation(slot).unwrap(),
                phase: Phase::Prefill,
                tokens: &prompts[slot],
                prompt_len: prompts[slot].len() as u32,
                selection: Selection::default(),
            })
            .collect();
        let mut outputs = Vec::new();
        e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
            upload_inputs(e, &m, p)
        })
        .unwrap();
        assert_eq!(outputs.len(), n);
        let diagnostic_slot = requests
            .iter()
            .find(|r| r.prompt_len == 7)
            .map(|r| r.slot as usize);
        archive_activations(&e, n, "prompt", diagnostic_slot);
        let fresh_slot = (n > 1).then_some(n - 1);
        let mut fresh_logits = Vec::new();
        if let Some(slot) = fresh_slot {
            let row = outputs.iter().position(|o| o.0 as usize == slot).unwrap();
            e.logits_row(row, &mut fresh_logits).unwrap();
        }
        draw(&mut e, &outputs, 0);
        let inputs: Vec<_> = (0..n)
            .map(|slot| [contract.input_vocab - 3 - slot as u32])
            .collect();
        let feeds: Vec<_> = (0..n)
            .rev()
            .filter(|&slot| Some(slot) != fresh_slot)
            .map(|slot| (slot, inputs[slot][0]))
            .collect();
        if e.handle_of("in.encoder_overlay_index").is_some() {
            let keep = vec![u32::MAX - 1; e.pf_max_rows()];
            e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&keep))
                .unwrap();
        }
        e.step_slots(&feeds, &mut Vec::new()).unwrap();
        archive_activations(&e, n, "ordinary", diagnostic_slot);
        let positions: Vec<_> = feeds
            .iter()
            .map(|&(slot, _)| (slot, prompts[slot].len()))
            .collect();
        let ordinary_kv = kv_vectors(&e, &blob, &positions);
        let mut reference = vec![Vec::new(); n];
        for &(slot, _) in &feeds {
            e.logits_row(slot, &mut reference[slot]).unwrap();
            e.rewind_slot(slot, prompts[slot].len()).unwrap();
        }
        // All-decode batches use prefill attention; mixed batches arm the decode riders.
        // Keep the terminal row count fixed so the control uses the same head geometry.
        let control: Vec<_> = (0..n)
            .rev()
            .map(|slot| Request {
                id: slot as u32,
                slot: slot as u32,
                state_slot: slot as u32,
                generation: e.slot_generation(slot).unwrap(),
                phase: Phase::Decode,
                tokens: &inputs[slot],
                prompt_len: prompts[slot].len() as u32,
                selection: Selection::default(),
            })
            .collect();
        e.token_batch_step_staged(&control, &mut outputs, None, &mut |_| {}, &mut |e, p| {
            upload_inputs(e, &m, p)
        })
        .unwrap();
        archive_activations(&e, n, "control", diagnostic_slot);
        let control_kv = kv_vectors(&e, &blob, &positions);
        let mut packed_reference = vec![Vec::new(); n];
        for (row, &(slot, _)) in outputs.iter().enumerate() {
            e.logits_row(row, &mut packed_reference[slot as usize])
                .unwrap();
            e.rewind_slot(slot as usize, prompts[slot as usize].len())
                .unwrap();
        }
        if let Some(slot) = fresh_slot {
            e.retire_slot(slot, false);
            assert_eq!(
                e.admit_packed_slot(slot, &prompts[slot], prompts[slot].len() + 8)
                    .unwrap(),
                Some(0)
            );
            bind(&mut e, slot);
        }
        let requests: Vec<_> = (0..n)
            .rev()
            .map(|slot| Request {
                id: slot as u32,
                slot: slot as u32,
                state_slot: slot as u32,
                generation: e.slot_generation(slot).unwrap(),
                phase: if Some(slot) == fresh_slot {
                    Phase::Prefill
                } else {
                    Phase::Decode
                },
                tokens: if Some(slot) == fresh_slot {
                    &prompts[slot]
                } else {
                    &inputs[slot]
                },
                prompt_len: prompts[slot].len() as u32,
                selection: Selection::default(),
            })
            .collect();
        e.token_batch_step_staged(&requests, &mut outputs, None, &mut |_| {}, &mut |e, p| {
            upload_inputs(e, &m, p)
        })
        .unwrap();
        assert_eq!(outputs.len(), n);
        archive_activations(&e, n, "mixed", diagnostic_slot);
        let mixed_kv = kv_vectors(&e, &blob, &positions);
        for (((name, ordinary), (control_name, control)), (mixed_name, mixed)) in
            ordinary_kv.iter().zip(&control_kv).zip(&mixed_kv)
        {
            assert_eq!((name, name), (control_name, mixed_name));
            let (ordinary_rel, ordinary_max) = kv_difference(ordinary, mixed);
            let (control_rel, control_max) = kv_difference(control, mixed);
            eprintln!("KV c{n} {name}: ordinary rel={ordinary_rel:.8} max={ordinary_max:.8}, prefill-control rel={control_rel:.8} max={control_max:.8}");
        }
        let mut seen = vec![false; n];
        let mut max_kl = 0f64;
        let mut max_packed_kl = 0f64;
        let mut mixed_logits = Vec::new();
        for (row, &(owner, _)) in outputs.iter().enumerate() {
            let slot = owner as usize;
            assert!(slot < n && !std::mem::replace(&mut seen[slot], true));
            let is_fresh = Some(slot) == fresh_slot;
            assert_eq!(
                e.pos[slot] as usize,
                prompts[slot].len() + usize::from(!is_fresh)
            );
            let mut actual = Vec::new();
            e.logits_row(row, &mut actual).unwrap();
            let reference = if is_fresh {
                &fresh_logits
            } else {
                &reference[slot]
            };
            let kl = logit_kl(reference, &actual);
            max_kl = max_kl.max(kl);
            if !is_fresh {
                let packed_kl = logit_kl(&packed_reference[slot], &actual);
                max_packed_kl = max_packed_kl.max(packed_kl);
                if packed_kl >= 1e-3 {
                    eprintln!("concurrency {n}, slot {slot}: packed-control KL={packed_kl}");
                    discrepancies.push(("packed", n, slot, packed_kl));
                }
            }
            if kl >= 1e-3 {
                let reference_top = crate::text::sample::argmax(reference) as usize;
                let actual_top = crate::text::sample::argmax(&actual) as usize;
                eprintln!("concurrency {n}, slot {slot}: KL={kl}, fresh={is_fresh}, top1 reference={reference_top} actual={actual_top}, reference gap={}", reference[reference_top] - reference[actual_top]);
                discrepancies.push(("ordinary", n, slot, kl));
            }
            mixed_logits.push((slot, actual));
        }
        if let Some(directory) = crate::config::RuntimeConfig::get().nv.nv_dump_kv.as_ref() {
            let data = serde_json::json!({"concurrency": n, "prompt_ids": &prompts,
                "ordinary": &reference, "control": &packed_reference, "mixed": &mixed_logits,
                "fresh_slot": fresh_slot, "fresh_prompt": &fresh_logits});
            serde_json::to_writer(
                std::fs::File::create(directory.join(format!("c{n}-logits.json"))).unwrap(),
                &data,
            )
            .unwrap();
        }
        eprintln!(
            "concurrency {n}: all physical owners/frontiers correct, max ordinary KL={max_kl}, max packed-control KL={max_packed_kl}"
        );
        let rows: Vec<_> = outputs
            .iter()
            .map(|&(owner, _)| FishRow {
                slot: owner as usize,
                hidden_row: owner,
                seed: 193 + u64::from(owner),
                step: u32::from(Some(owner as usize) != fresh_slot),
                temperature: 1.0,
                top_p: 1.0,
                top_k: 1,
                ras_temperature: 1.0,
                ras_top_p: 1.0,
                repetition_penalty: 1.0,
                cfg: None,
            })
            .collect();
        let owners: Vec<_> = outputs.iter().map(|o| o.0).collect();
        let mut codes = Vec::new();
        e.fish_depth_step_compact(&rows, &owners, &mut codes)
            .unwrap();
        assert_eq!(codes.len(), n * m.codebooks as usize);
        assert!(codes
            .chunks_exact(m.codebooks as usize)
            .all(|c| c[0] <= m.stop_class));
        for slot in 0..n {
            e.retire_slot(slot, false);
        }
    }
    assert!(
        discrepancies.is_empty(),
        "ordinary vs packed decode KL >= 1e-3: {discrepancies:?}"
    );
}
