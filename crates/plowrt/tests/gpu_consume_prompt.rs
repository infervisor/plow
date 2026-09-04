//! Decode-loop prompt consumption (`GpuEngine::consume_prompt`) must produce
//! EXACTLY the first token and follow-on greedy tokens that per-token
//! `step_slots` produces from the same prompt. One host sync vs L must not
//! change numerics.
//!
//! Gated on `PLOW_GPU_TEST=1` + assets (`PLOW_GPU_ASSETS`). Skips silently.

#![cfg(feature = "cuda")]

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
