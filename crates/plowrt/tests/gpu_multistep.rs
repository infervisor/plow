//! Bounded device multi-step token identity (plan stage 5). A K-token quantum
//! via `GpuEngine::multi_step` (one host sync; device-owned pos/kvlen advanced
//! by `plow_advance` between decode launches) must produce EXACTLY the tokens
//! that K individual greedy `step_slots` calls produce from the same state.
//!
//! Gated on `PLOW_GPU_TEST=1` + assets (`PLOW_GPU_ASSETS`, default the b4 dir).
//! Builds the sampler cubin (carries `plow_advance`) with nvcc. Skips silently.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::exec::gpu::GpuEngine;

const SMP_SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");

#[test]
fn multi_step_matches_single_step_greedy() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    let ckpt = assets.join("checkpoint");

    // Build the sampler cubin (has plow_advance).
    let dir = std::env::temp_dir().join(format!("plowrt-mstep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("sample_sm120.cu");
    let cubin = dir.join("sample_sm120.cubin");
    std::fs::write(&src, SMP_SRC).unwrap();
    let out = std::process::Command::new("/usr/local/cuda/bin/nvcc")
        .env_clear()
        .env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin")
        .args(["-arch=native", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc");
    assert!(
        out.status.success(),
        "nvcc: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    const K: usize = 8;
    std::env::set_var("PLOW_MULTISTEP", K.to_string());
    std::env::set_var("PLOW_NV_CUBIN_SAMPLE", &cubin);

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("engine load");
    assert_eq!(
        e.multistep_quantum(),
        Some(K),
        "multi-step did not enable (needs a dynamic-kvrow decode cubin)"
    );

    let prompt: Vec<u32> = (0..32u32).map(|i| 100 + i).collect();
    let n_gen = 2 * K; // two full quanta

    // Reference: single greedy steps.
    let single: Vec<u32> = {
        e.begin_slot(0, prompt.len() + n_gen + 1).expect("begin");
        let first = e.prefill_slot(0, &prompt).expect("prefill");
        let mut out = Vec::new();
        let mut toks = Vec::new();
        let mut last = first;
        for _ in 0..n_gen {
            e.step_slots(&[(0, last)], &mut toks).expect("step");
            last = toks[0];
            out.push(last);
        }
        out
    };

    // Multi-step: two K-quanta from the same reset state.
    let multi: Vec<u32> = {
        e.begin_slot(0, prompt.len() + n_gen + 1).expect("begin");
        let first = e.prefill_slot(0, &prompt).expect("prefill");
        let mut out = Vec::new();
        let mut buf = Vec::new();
        let mut last = first;
        for _ in 0..(n_gen / K) {
            let k = e.multi_step(&[(0, last)], &mut buf).expect("multi_step");
            assert_eq!(k, K);
            assert_eq!(buf.len(), K, "one fed row → K tokens");
            out.extend_from_slice(&buf);
            last = *out.last().unwrap();
        }
        out
    };

    assert_eq!(
        multi, single,
        "multi-step ({K}/quantum) diverged from single-step greedy:\n multi={multi:?}\nsingle={single:?}"
    );
    eprintln!(
        "multi-step OK: {n_gen} tokens identical single vs {K}-quantum; first {:?}",
        &single[..single.len().min(8)]
    );

    let block = std::fs::read(assets.join("block.json"))
        .ok()
        .map(|bytes| serde_json::from_slice::<plow_asset::BlockDescriptor>(&bytes).unwrap());
    let mut sparse = vec![0];
    for slot in [3, 15] {
        if slot < e.batch() {
            sparse.push(slot);
        }
    }
    sparse.reverse();
    for (prompt_len, requested) in [(32, 3), (32, 1), (e.max_ctx() - 4, K)] {
        let steps = requested.min(e.max_ctx() - prompt_len);
        let run = |e: &mut GpuEngine, multi: bool| {
            for slot in 0..e.batch() {
                e.begin_slot(slot, e.max_ctx()).unwrap();
            }
            let mut feeds = Vec::new();
            for &slot in &sparse {
                if let Some(block) = &block {
                    let input: Vec<_> = (0..e.tensor_bytes("act.x").unwrap() as usize / 2)
                        .map(|i| ((i * 13 + slot * 17) % 251) as f32 / 251.0 - 0.5)
                        .collect();
                    e.upload_activation("act.x", &input).unwrap();
                    assert!(block.hidden > 0);
                }
                let prompt: Vec<_> = (0..prompt_len)
                    .map(|i| 100 + ((i + slot * 17) % 1000) as u32)
                    .collect();
                feeds.push((slot, e.prefill_slot(slot, &prompt).unwrap()));
            }
            let mut tokens = Vec::new();
            if multi {
                assert_eq!(
                    e.multi_step_at_most(&feeds, requested, &mut tokens)
                        .unwrap(),
                    steps
                );
            } else {
                tokens.resize(feeds.len() * steps, 0);
                let mut one = Vec::new();
                for step in 0..steps {
                    e.step_slots(&feeds, &mut one).unwrap();
                    for (row, feed) in feeds.iter_mut().enumerate() {
                        tokens[row * steps + step] = one[row];
                        feed.1 = one[row];
                    }
                }
            }
            for &slot in &sparse {
                assert_eq!(e.attach_prompt(slot, &[]).unwrap(), prompt_len + steps);
            }
            let mut state = Vec::new();
            if let Some(block) = &block {
                let row_bytes = block.hidden as usize * 2;
                for &slot in &sparse {
                    let mut bytes = vec![0; row_bytes];
                    e.read_tensor_range("act.x", (slot * row_bytes) as u64, &mut bytes)
                        .unwrap();
                    assert!(bytes.chunks_exact(2).all(|x| {
                        f32::from_bits(u32::from(u16::from_le_bytes([x[0], x[1]])) << 16)
                            .is_finite()
                    }));
                    state.extend(bytes);
                }
                let heads = block.dims.kv_heads.unwrap() as usize;
                let kv_row_bytes = block.dims.head_dim.unwrap() as usize * 2;
                for carried in &block.carried_state {
                    assert_eq!(carried.role, "kv", "requires a direct-KV block");
                    for name in &carried.tensors {
                        let slot_bytes = e.tensor_bytes(name).unwrap() as usize / e.batch();
                        let head_bytes = slot_bytes / heads;
                        let live_bytes = ((prompt_len + steps) * kv_row_bytes).min(head_bytes);
                        for &slot in &sparse {
                            for head in 0..heads {
                                let offset = (slot * slot_bytes + head * head_bytes) as u64;
                                let mut bytes = vec![0; live_bytes];
                                e.read_tensor_range(name, offset, &mut bytes).unwrap();
                                state.extend(bytes);
                            }
                        }
                    }
                }
            } else {
                for &slot in &sparse {
                    let mut logits = Vec::new();
                    e.logits_row(slot, &mut logits).unwrap();
                    assert!(logits.iter().all(|v| v.is_finite()));
                    state.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
                }
            }
            (tokens, state)
        };
        let expected = run(&mut e, false);
        let actual = run(&mut e, true);
        assert_eq!(actual.0, expected.0, "sparse token ring");
        assert_eq!(actual.1.len(), expected.1.len(), "state extent");
        assert_eq!(
            actual.1.iter().zip(&expected.1).position(|(a, b)| a != b),
            None,
            "first state mismatch: sparse {sparse:?}, prompt={prompt_len}, steps={steps}"
        );
        eprintln!("multistep tail exact: slots={sparse:?}, prompt={prompt_len}, requested={requested}, steps={steps}");
    }

    std::env::remove_var("PLOW_MULTISTEP");
    std::env::remove_var("PLOW_NV_CUBIN_SAMPLE");
}
