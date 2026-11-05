use super::*;
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
                    index[span.row0 as usize + r] = (overlays.len() / hidden) as u32;
                    overlays.extend((0..hidden).map(|j| ((j % 17) as f32 - 8.0) * 0.001));
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
    let mut codes = Vec::new();
    e.fish_depth_step_compact(&rows, &owners, &mut codes)
        .unwrap();
    let m = &e.fish.as_ref().unwrap().manifest;
    assert_eq!(codes.len(), rows.len() * m.codebooks as usize);
    assert!(codes
        .chunks_exact(m.codebooks as usize)
        .all(|c| c[0] <= m.stop_class));
}

#[test]
#[ignore = "GPU correctness only; SPEECH_BATCH_TEST_ASSETS, PLOW_VMM_PREFIX=1; run through gpuq"]
fn speech_mixed_batch_maps_frames_and_reuses_only_matching_content() {
    let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
    assert!(e.token_batch_enabled() && e.packed_depth_enabled());
    assert!(e.vmm_prefix_enabled());
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

    let mut a = vec![contract.input_vocab - 1; 41];
    let mut b = vec![contract.input_vocab - 2; 37];
    if let Some(token) = contract.decode_token {
        a[3] = token;
        b[3] = token;
    }
    let keys_a: Vec<u64> = (0..a.len()).map(|i| i as u64 + 900).collect();
    let keys_b: Vec<u64> = (0..b.len()).map(|i| i as u64 + 1900).collect();
    for (slot, prompt, keys) in [(3, &a, &keys_a), (0, &b, &keys_b)] {
        assert!(e.stage_speech_prefix(slot, keys));
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
    draw(&mut e, &outputs, 0);

    // A continuing decode row rides beside a new prompt, through the hd128 split-attention arm.
    e.retire_slot(0, false);
    assert!(e.stage_speech_prefix(0, &keys_b));
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
    e.rewind_slot(3, a.len()).unwrap();
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
    assert_eq!((e.pos[3], e.pos[0]), (42, 37));
    let mut actual = Vec::new();
    e.logits_row(0, &mut actual).unwrap();
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
    assert_eq!(
        crate::text::sample::argmax(&reference),
        crate::text::sample::argmax(&actual)
    );
    draw(&mut e, &outputs, 1);
    e.retire_slot(0, false);
    let mut changed = keys_b.clone();
    changed[0] ^= 1;
    assert!(e.stage_speech_prefix(0, &changed));
    assert_eq!(e.admit_packed_slot(0, &b, b.len() + 8).unwrap(), Some(0));
    e.retire_slot(0, false);
    e.begin_slot(0, b.len() + 8).unwrap();
    e.set_prefix_attach(0, true);
    assert_eq!(
        e.attach_prompt(0, &b).unwrap(),
        0,
        "raw tokens aliased speech identities"
    );
    let before = e.pos.clone();
    assert!(
        e.token_batch_step(&requests, &mut outputs).is_err(),
        "unstaged speech was accepted"
    );
    assert_eq!(e.pos, before);
}
