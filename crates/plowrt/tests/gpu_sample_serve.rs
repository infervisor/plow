//! Device-sampler serve-path integration (plan stage 4). Loads a real engine
//! with `PLOW_DEV_SAMPLE=1` and drives `step_slots_sampled`, checking:
//!   1. the sampler is actually enabled,
//!   2. a `temp==0` spec reproduces the greedy `step_slots` tokens exactly
//!      (the device sampler's greedy branch == ARGMAX_FIN),
//!   3. stochastic decode is deterministic for a fixed per-step rng and stays
//!      in-vocab (the token was written device-side into in.ids, no host D2H).
//!
//! Gated on `PLOW_GPU_TEST=1` + assets (`PLOW_GPU_ASSETS`, default
//! /root/gpu-assets-b4/b4). Builds the sampler cubin with nvcc. Skips silently.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::exec::gpu::{DevSample, GpuEngine};

const SMP_SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");

/// Deterministic per-step uniform in [0,1) for the test (no model tokenizer
/// needed — reproducibility is about same-input→same-output).
fn rng_at(step: usize) -> f32 {
    (((step as u64).wrapping_mul(2654435761) >> 11) & 0xffff) as f32 / 65536.0
}

#[test]
fn device_sampler_serve_integration() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    let ckpt = assets.join("checkpoint");

    // Build the sampler cubin.
    let dir = std::env::temp_dir().join(format!("plowrt-smpserve-{}", std::process::id()));
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

    // Enable device sampling for this engine.
    std::env::set_var("PLOW_DEV_SAMPLE", "1");
    std::env::set_var("PLOW_NV_CUBIN_SAMPLE", &cubin);

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("engine load");
    assert!(e.dev_sample_enabled(), "device sampler did not load");

    let cap = e.batch();
    let prompt: Vec<u32> = (0..32u32).map(|i| 100 + i).collect();
    let max_new = 24usize;

    // Helper: prefill slot 0 then decode `max_new` tokens under `specs_fn`
    // (None = plain greedy step_slots).
    fn run(
        e: &mut GpuEngine,
        prompt: &[u32],
        max_new: usize,
        specs_fn: Option<&dyn Fn(usize) -> Vec<DevSample>>,
    ) -> Vec<u32> {
        e.begin_slot(0, prompt.len() + max_new).expect("begin");
        let first = e.prefill_slot(0, prompt).expect("prefill");
        let mut out = vec![first];
        let mut toks = Vec::new();
        let mut last = first;
        for step in 0..max_new - 1 {
            match specs_fn {
                Some(f) => {
                    let specs = f(step);
                    e.step_slots_sampled(&[(0, last)], Some(&specs), &mut toks)
                        .expect("step");
                }
                None => e.step_slots(&[(0, last)], &mut toks).expect("step"),
            }
            last = toks[0];
            out.push(last);
        }
        out
    }

    // 1. Greedy spec (temp=0) must equal the plain greedy step_slots run.
    let greedy_specs = move |_step: usize| vec![DevSample::greedy(); cap];
    let dev_greedy = run(&mut e, &prompt, max_new, Some(&greedy_specs));
    let plain_greedy = run(&mut e, &prompt, max_new, None);
    assert_eq!(
        dev_greedy, plain_greedy,
        "device sampler temp=0 must reproduce greedy ARGMAX_FIN tokens"
    );

    // 2. Stochastic determinism: same fixed per-step rng twice → same tokens.
    // A high temperature over a wide top_k flattens even a confident synthetic
    // prompt enough that sampling should diverge from greedy (the distribution
    // correctness itself is the standalone gpu_sample test's job, at TVD<=5e-4).
    let stoch_specs = move |step: usize| {
        let mut v = vec![DevSample::greedy(); cap];
        v[0] = DevSample {
            temp: 30.0,
            top_k: 512,
            top_p: 1.0,
            min_p: 0.0,
            rng01: rng_at(step),
        };
        v
    };
    let s1 = run(&mut e, &prompt, max_new, Some(&stoch_specs));
    let s2 = run(&mut e, &prompt, max_new, Some(&stoch_specs));
    assert_eq!(
        s1, s2,
        "stochastic device sampling not deterministic for fixed rng"
    );

    let vocab = e.vocab() as u32;
    assert!(s1.iter().all(|&t| t < vocab), "token out of vocab range");
    // Divergence from greedy is expected but not guaranteed on a pathologically
    // confident synthetic prompt — report rather than hard-fail (the sampler's
    // distribution is gated exactly by the standalone gpu_sample test).
    if s1 == dev_greedy {
        eprintln!(
            "NOTE: high-temp sampling still matched greedy — synthetic prompt is \
             degenerate (one dominant token); sampler correctness is covered by gpu_sample"
        );
    }

    eprintln!(
        "dev-sample serve OK: greedy≡ARGMAX_FIN ({} toks), stochastic deterministic, in-vocab; \
         diverged_from_greedy={}, first stochastic tokens {:?}",
        dev_greedy.len(),
        s1 != dev_greedy,
        &s1[..s1.len().min(6)]
    );

    std::env::remove_var("PLOW_DEV_SAMPLE");
    std::env::remove_var("PLOW_NV_CUBIN_SAMPLE");
}
