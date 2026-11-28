//! Fish S2 Pro frame generation on one slot: prefill a prompt, then per frame run the depth stage
//! (backbone class + codebooks) and feed the frame back. Writes the codes (`[frames][codebooks]`
//! u32, the stop frame excluded) for the codec gate.
//!
//!   fish_gen ASSETS CKPT PROMPT.u32 OUT.u32 [--prompt-codes CODES.u32] [--frames N] [--seed S]
//!            [--temperature T] [--top-p P] [--top-k K] [--slots N [--depth-sweep STRIDE]]
//!            [--unread-backbone] [--timing-json FILE] [--first-frame-repeats N]
//!
//! PROMPT.u32 holds token ids; prompt frame rows (reference audio) take their codes from
//! `--prompt-codes` (`[prompt rows][codebooks]`). `--top-k 1` is greedy and matches the goldens.
//! `--slots N` instead times N slots stepped together for `--frames` steps (seeds `S..S+N`; a stop
//! frame is fed as class 0 so the width stays N) and prints the depth and backbone ms per step.
use std::path::PathBuf;
use std::sync::Arc;

use plowrt::exec::gpu::{FishRow, GpuEngine};

fn read_u32(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Fish keeps temperature and top_p as bf16 tensors (0.7 -> 0.69921875).
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn bind_ordering_screen(
    e: &mut GpuEngine,
    be: &plowrt::device::cuda::CudaBackend,
    prompt: &[u32],
    mut plan: plowrt::tts::maya2::Plan,
    seed: u64,
    same_memory: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let null = e.maya2_null_memory().ok_or("missing speaker memory")?;
    let old: Arc<[f32]> = null.iter().map(|v| v * -3.0 + 0.125).collect();
    let new: Arc<[f32]> = null.iter().map(|v| v * 0.25 - 0.0625).collect();
    let mut poisoned_plan = plan;
    if same_memory {
        poisoned_plan.min_frames = 0;
        poisoned_plan.cb0.stop_p = 0.0;
        plan.min_frames = u32::MAX;
    }
    let info = e.fish_info()?;
    let row = FishRow {
        slot: 0,
        hidden_row: 0,
        seed,
        step: 0,
        temperature: bf16(0.7),
        top_p: bf16(0.7),
        top_k: 30,
        ras_temperature: bf16(1.0),
        ras_top_p: bf16(0.9),
        repetition_penalty: 1.0,
        cfg: None,
    };
    let mut outputs = Vec::new();
    let mut activation_reference = None;
    let mut poison_changed_codes = false;
    let mut poison_changed_activations = false;
    let completion = be.event_create(false)?;
    let mut pending_binds = 0;
    for iteration in 0..16 {
        let mut pair = Vec::new();
        for (fenced, poison) in [
            (iteration % 2 == 0, false),
            (iteration % 2 != 0, false),
            (true, true),
        ] {
            e.begin_slot(0, prompt.len() + 3)?;
            e.fish_reset_slot(0)?;
            e.maya2_bind(
                0,
                if same_memory { poisoned_plan } else { plan },
                Some(if same_memory { &new } else { &old }),
            )?;
            be.synchronize()?;
            e.prefill_slot(0, prompt)?;
            e.fish_logits_to_slot(0)?;
            let mut codes = Vec::new();
            e.fish_depth_step(
                &[FishRow {
                    hidden_row: e.prefill_terminal_row(),
                    ..row
                }],
                &mut codes,
            )?;
            if same_memory && codes[0] != info.stop_class {
                return Err(
                    "inconclusive bind ordering screen: controller poison did not stop".into(),
                );
            }
            let initial_codes = codes.clone();
            let class = if codes[0] == info.stop_class {
                0
            } else {
                codes[0]
            };
            e.step_slots_unread(&[(0, info.semantic_base + class)], &mut Vec::new())?;
            e.record_stream_event(&completion)?;
            if !fenced && !be.event_query(&completion)? {
                pending_binds += 1;
            }
            if fenced {
                e.sync_stream()?;
            }
            e.maya2_bind(
                0,
                if same_memory && poison {
                    poisoned_plan
                } else {
                    plan
                },
                Some(if same_memory || !poison { &new } else { &old }),
            )?;
            if fenced {
                be.synchronize()?;
            }
            codes.clear();
            e.fish_depth_step(&[row], &mut codes)?;
            if same_memory && (codes[0] == info.stop_class) != poison {
                return Err(format!("controller bind stop mismatch: fenced={fenced}, poison={poison}, codes={codes:?}").into());
            }
            let mut activations = vec![0u8; 1024 * 4];
            e.read_tensor_range("act.depth.x", 0, &mut activations)?;
            let layout = e.depth_plan_describe().ok_or("missing depth plan")?;
            let layout = layout.split(" cost_us ").next().unwrap().to_owned();
            pair.push((fenced, codes, activations, layout, initial_codes));
        }
        if pair
            .iter()
            .any(|arm| arm.3 != pair[0].3 || arm.4 != pair[0].4)
        {
            return Err(
                "inconclusive bind ordering screen: initialization or depth capacity differs"
                    .into(),
            );
        }
        if pair[0].1 != pair[1].1 || pair[0].2 != pair[1].2 {
            return Err(format!("bind ordering mismatch at iteration {iteration}").into());
        }
        if outputs.first().is_some_and(|v| v != &pair[0].1)
            || activation_reference
                .as_ref()
                .is_some_and(|v| v != &pair[0].2)
        {
            return Err(format!(
                "inconclusive bind ordering screen: reference changed at iteration {iteration}"
            )
            .into());
        }
        poison_changed_codes |= pair[0].1 != pair[2].1;
        poison_changed_activations |= pair[0].2 != pair[2].2;
        activation_reference = Some(pair[0].2.clone());
        outputs.push(pair.remove(0).1);
    }
    if !poison_changed_codes && !poison_changed_activations {
        return Err(
            "inconclusive bind ordering screen: poison did not change observed outputs".into(),
        );
    }
    if pending_binds == 0 {
        return Err("inconclusive bind ordering screen: no pending backbone at bind".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "bind_ordering": "pass",
            "case": if same_memory { "same_speaker_state_reset" } else { "changed_speaker" },
            "fenced_unfenced_pairs": 16,
            "observed_pending_binds": pending_binds,
            "poison_changed_codes": poison_changed_codes,
            "poison_changed_activations": poison_changed_activations,
            "speaker_bytes": new.len() * 4,
            "depth_plan": e.depth_plan_describe(),
            "codes": outputs.first(),
        })
    );
    Ok(())
}

/// `--depth-sweep STRIDE` (with `--slots N`): the depth program of the width covering N slots
/// timed at every STRIDE-th instruction cap (median of 20 launches, JSON lines
/// `{"cap":..,"op":..,"blocks":..,"us":..}`, `op` the instruction cap - 1 adds), so the marginal
/// cost of instruction i is us(cap = i + 1) - us(cap = i).
fn depth_sweep(
    e: &mut GpuEngine,
    assets: &std::path::Path,
    rows: &[(FishRow, u32)],
    n: usize,
    stride: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read(assets.join("model.pkt"))?;
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw)?;
    let meta = blob
        .reserved_metadata(&raw, plow_asset::fish_depth::SECTION)?
        .ok_or("packet has no depth stage")?;
    let manifest: plow_asset::fish_depth::Manifest = serde_json::from_slice(meta)?;
    let bytes = blob
        .reserved_metadata(&raw, plow_asset::fish_depth::PROGRAMS_SECTION)?
        .ok_or("depth programs section missing")?;
    let section = plow_asset::aux_program::parse(bytes, blob.n_cu, blob.tensors.len())?;
    let (k, _) = manifest
        .rows
        .iter()
        .enumerate()
        .filter(|(_, &r)| r as usize >= n)
        .min_by_key(|(_, &r)| r)
        .ok_or("no depth program covers the slots")?;
    let insts = &section.programs[k].insts;
    let step: Vec<FishRow> = rows
        .iter()
        .map(|(r, _)| FishRow {
            hidden_row: r.slot as u32,
            ..*r
        })
        .collect();
    let mut codes = Vec::new();
    for cap in (0..=insts.len()).step_by(stride) {
        e.set_depth_debug_max_inst(cap as u32)?;
        let us: Result<Vec<f64>, _> = (0..23)
            .map(|_| {
                codes.clear();
                let t = std::time::Instant::now();
                e.fish_depth_step(&step, &mut codes)
                    .map(|_| t.elapsed().as_secs_f64() * 1e6)
            })
            .collect();
        let mut us = match us {
            Ok(us) => us,
            Err(err) => {
                e.set_depth_debug_max_inst(u32::MAX)?;
                return Err(err.into());
            }
        };
        us.drain(..3);
        us.sort_by(f64::total_cmp);
        let (op, blocks) = cap.checked_sub(1).map_or((String::new(), 0), |i| {
            let op = packet::dev::DevOp::from_u16(insts[i].op)
                .map_or_else(|| insts[i].op.to_string(), |o| format!("{o:?}"));
            (op, insts[i].blocks)
        });
        println!(
            "{{\"rows\":{},\"cap\":{cap},\"op\":\"{op}\",\"blocks\":{blocks},\"us\":{:.2}}}",
            manifest.rows[k],
            us[us.len() / 2]
        );
    }
    e.set_depth_debug_max_inst(u32::MAX)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: fish_gen ASSETS CKPT PROMPT.u32 OUT.u32 [--prompt-codes F] [--frames N] [--seed S] [--temperature T] [--top-p P] [--top-k K]");
        std::process::exit(2);
    }
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let frames: usize = opt("--frames").map_or(512, |v| v.parse().unwrap());
    let seed: u64 = opt("--seed").map_or(42, |v| v.parse().unwrap());
    let temperature = bf16(opt("--temperature").map_or(0.7, |v| v.parse().unwrap()));
    let top_p = bf16(opt("--top-p").map_or(0.7, |v| v.parse().unwrap()));
    let top_k: u32 = opt("--top-k").map_or(30, |v| v.parse().unwrap());
    let unread = a.iter().any(|v| v == "--unread-backbone");
    let first_repeats: usize = opt("--first-frame-repeats").map_or(1, |v| v.parse().unwrap());
    if first_repeats == 0 || (first_repeats != 1 && opt("--slots").as_deref() != Some("1")) {
        return Err("--first-frame-repeats requires a positive count and --slots 1".into());
    }
    let assets = PathBuf::from(&a[1]);
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let bind_screen = a.iter().any(|v| v == "--bind-ordering-screen");
    if bind_screen {
        eprintln!("bind ordering: loading model");
    }
    let mut e = GpuEngine::load(be.clone(), &assets, &PathBuf::from(&a[2]))?;
    if bind_screen {
        eprintln!("bind ordering: model ready");
    }
    let info = e.fish_info()?;
    let prompt = read_u32(&a[3]);
    // A Maya 2 depth stage needs its host cb0 controller bound per slot (the checkpoint's
    // `maya2_controller.json`, as `plowrt serve` binds it); the null speaker's depth memory.
    let maya2 = std::fs::read_to_string(PathBuf::from(&a[2]).join("maya2_controller.json"))
        .ok()
        .map(|s| plowrt::tts::maya2::Controller::parse(&s))
        .transpose()?
        .map(|c| c.plan("fish_gen timing prompt", "en", Some(frames), c.cb0, seed));
    if let Some(p) = opt("--prompt-codes") {
        e.fish_prompt_codes(&read_u32(&p))?;
    }
    if bind_screen {
        let plan = maya2.ok_or("missing controller plan")?;
        bind_ordering_screen(&mut e, &be, &prompt, plan, seed, true)?;
        return bind_ordering_screen(&mut e, &be, &prompt, plan, seed, false);
    }
    if let Some(n) = opt("--slots").map(|v| v.parse::<usize>().unwrap()) {
        let mut rows = Vec::new();
        let mut codes = Vec::new();
        // `--same-streams P`: every slot runs the same stream, with speaker memory `slot % P`, so
        // slot `s` must emit slot `s % P`'s codes whatever windows the depth plan runs it in.
        // `--memory-step F`: speaker memory `j` is the null memory scaled by `1 + F * j`.
        let memory_step: f32 = opt("--memory-step").map_or(0.25, |v| v.parse().unwrap());
        let period = opt("--same-streams").map(|v| v.parse::<usize>().unwrap().max(1));
        let memories: Vec<std::sync::Arc<[f32]>> = match (period, e.maya2_null_memory()) {
            (Some(p), Some(null)) => (0..p)
                .map(|j| {
                    null.iter()
                        .map(|v| v * (1.0 + memory_step * j as f32))
                        .collect()
                })
                .collect(),
            _ => Vec::new(),
        };
        let (mut same, mut distinct) = (0usize, 0usize);
        let (mut off_by_memory, mut first_off) = (vec![0usize; period.unwrap_or(0)], None);
        // `--slot-stride K`: the N streams sit in slots 0, K, 2K, ..
        let stride = opt("--slot-stride").map_or(1, |v| v.parse::<usize>().unwrap().max(1));
        let mut first_ms = Vec::new();
        for (index, slot) in (0..n).map(|i| (i, i * stride)) {
            for repeat in 0..first_repeats {
                e.begin_slot(slot, prompt.len() + frames + 1)?;
                e.fish_reset_slot(slot)?;
                if let Some(p) = maya2 {
                    e.maya2_bind(slot, p, memories.get(index % memories.len().max(1)))?;
                }
                let first = std::time::Instant::now();
                e.prefill_slot(slot, &prompt)?;
                e.fish_logits_to_slot(slot)?;
                let row = FishRow {
                    slot,
                    hidden_row: e.prefill_terminal_row(),
                    seed: seed + if period.is_some() { 0 } else { index as u64 },
                    step: 0,
                    temperature,
                    top_p,
                    top_k,
                    ras_temperature: bf16(1.0),
                    ras_top_p: bf16(0.9),
                    repetition_penalty: 1.0,
                    cfg: None,
                };
                codes.clear();
                e.fish_depth_step(&[row], &mut codes)?;
                first_ms.push(first.elapsed().as_secs_f64() * 1e3);
                if repeat + 1 == first_repeats {
                    rows.push((row, codes[0]));
                }
            }
        }
        let mut toks = Vec::new();
        let (mut depth, mut backbone) = (0.0f64, 0.0f64);
        // `--codes-out FILE`: every frame's codes of every stream, `[frames][N][codebooks]` u32.
        let mut all_codes: Vec<u32> = Vec::new();
        let mut frame_ms = Vec::new();
        for frame in 0..frames {
            let tick = std::time::Instant::now();
            let feeds: Vec<(usize, u32)> = rows
                .iter()
                .map(|(r, class)| {
                    (
                        r.slot,
                        info.semantic_base + if *class == info.stop_class { 0 } else { *class },
                    )
                })
                .collect();
            let t = std::time::Instant::now();
            if unread {
                e.step_slots_unread(&feeds, &mut toks)?;
            } else {
                e.step_slots(&feeds, &mut toks)?;
            }
            backbone += t.elapsed().as_secs_f64();
            let step: Vec<FishRow> = rows
                .iter()
                .map(|(r, _)| FishRow {
                    step: r.step + 1,
                    hidden_row: r.slot as u32,
                    ..*r
                })
                .collect();
            codes.clear();
            let t = std::time::Instant::now();
            e.fish_depth_step(&step, &mut codes)?;
            depth += t.elapsed().as_secs_f64();
            frame_ms.push(tick.elapsed().as_secs_f64() * 1e3);
            all_codes.extend_from_slice(&codes);
            for (k, (r, class)) in rows.iter_mut().enumerate() {
                *r = step[k];
                *class = codes[k * info.codebooks];
            }
            if let Some(p) = period {
                let row = |k: usize| &codes[k * info.codebooks..(k + 1) * info.codebooks];
                same += (p..n).filter(|&k| row(k) != row(k % p)).count();
                for k in (p..n).filter(|&k| row(k) != row(k % p)) {
                    off_by_memory[k % p] += 1;
                    first_off.get_or_insert((
                        frame,
                        k,
                        (0..info.codebooks).find(|&c| row(k)[c] != row(k % p)[c]),
                    ));
                }
                distinct += (1..p.min(n)).filter(|&k| row(k) != row(0)).count();
            }
        }
        if let Some(path) = opt("--codes-out") {
            std::fs::write(path, bytemuck::cast_slice::<u32, u8>(&all_codes))?;
        }
        if let Some(path) = opt("--timing-json") {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "slots": n, "unread_backbone": unread, "first_frame_ms": first_ms,
                    "frame_ms": frame_ms, "backbone_submit_ms": backbone * 1e3,
                    "depth_complete_ms": depth * 1e3
                }))?,
            )?;
        }
        if let Some(plan) = e.depth_plan_describe() {
            println!("{plan}");
        }
        if let Some(p) = period {
            println!("same-streams period {p}: rows off their period row {same}, rows differing by memory {distinct}");
            println!("off by memory {off_by_memory:?} first (frame, row, level) {first_off:?}");
        }
        let f = frames.max(1) as f64;
        println!(
            "slots {n} backbone_ms {:.2} depth_ms {:.2} step_ms {:.2} frames_per_s {:.0}",
            backbone * 1e3 / f,
            depth * 1e3 / f,
            (backbone + depth) * 1e3 / f,
            n as f64 * f / (backbone + depth)
        );
        if let Some(stride) = opt("--depth-sweep").map(|v| v.parse::<usize>().unwrap().max(1)) {
            if maya2.is_some() {
                // Its depth programs launch from the speech module, which the cap does not reach.
                return Err(
                    "--depth-sweep: the instruction cap does not reach Maya 2 depth programs"
                        .into(),
                );
            }
            depth_sweep(&mut e, &assets, &rows, n, stride)?;
        }
        return Ok(());
    }
    e.begin_slot(0, prompt.len() + frames + 1)?;
    e.fish_reset_slot(0)?;
    if let Some(p) = maya2 {
        e.maya2_bind(0, p, None)?;
    }
    let t0 = std::time::Instant::now();
    e.prefill_slot(0, &prompt)?;
    e.fish_logits_to_slot(0)?;
    let ttft = t0.elapsed();
    let mut row = FishRow {
        slot: 0,
        hidden_row: e.prefill_terminal_row(),
        seed,
        step: 0,
        temperature,
        top_p,
        top_k,
        ras_temperature: bf16(1.0),
        repetition_penalty: 1.0,
        ras_top_p: bf16(0.9),
        cfg: None,
    };
    let mut out = Vec::new();
    let mut codes = Vec::new();
    let mut toks = Vec::new();
    let t1 = std::time::Instant::now();
    let mut first_frame = None;
    for _ in 0..frames {
        codes.clear();
        e.fish_depth_step(&[row], &mut codes)?;
        first_frame.get_or_insert_with(|| t0.elapsed());
        let class = codes[0];
        if class == info.stop_class {
            break;
        }
        out.extend_from_slice(&codes);
        e.step_slots(&[(0, info.semantic_base + class)], &mut toks)?;
        row.step += 1;
        row.hidden_row = 0;
    }
    let n = out.len() / info.codebooks;
    let gen = t1.elapsed();
    std::fs::write(&a[4], bytemuck::cast_slice(&out))?;
    let audio_s = n as f64 * 2048.0 / 44100.0;
    println!(
        "frames {n} prefill_ms {:.1} first_frame_ms {:.1} frame_ms {:.2} rtf {:.3}",
        ttft.as_secs_f64() * 1e3,
        first_frame.unwrap_or_default().as_secs_f64() * 1e3,
        gen.as_secs_f64() * 1e3 / n.max(1) as f64,
        gen.as_secs_f64() / audio_s.max(1e-9)
    );
    Ok(())
}
