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

    std::env::remove_var("PLOW_MULTISTEP");
    std::env::remove_var("PLOW_NV_CUBIN_SAMPLE");
}
