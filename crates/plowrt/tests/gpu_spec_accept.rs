//! `plow_spec_accept` (runtime/nvidia/sample_sm120.cu) against the host golden
//! `orch::speculative::accept_greedy` on random spans: every k in 0..=15, partial and full
//! acceptance, stop tokens at every emission position.
//!
//! Gated on `PLOW_GPU_TEST=1` (needs GPU + nvcc). Skips silently otherwise.

#![cfg(feature = "cuda")]

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::orch::speculative::{accept_greedy, SpecSpanDev, SpecVerdict, SPEC_KMAX};

const SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");

#[test]
fn device_accept_matches_host_golden() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + nvcc)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("plowrt-spec-accept-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("sample_sm120.cu");
    let cubin = dir.join("sample_sm120.cubin");
    std::fs::write(&src, SRC).unwrap();
    let nvcc = std::env::var("PLOW_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let mut cmd = std::process::Command::new(&nvcc);
    cmd.env_clear().env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin");
    if let Ok(flags) = std::env::var("NVCC_PREPEND_FLAGS") {
        cmd.env("NVCC_PREPEND_FLAGS", flags);
    }
    let out = cmd
        .args(["-arch=native", "-O3", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc not found");
    assert!(out.status.success(), "nvcc: {}", String::from_utf8_lossy(&out.stderr));
    let be = CudaBackend::new(0).expect("CUDA backend");
    let module = be.module_load(&std::fs::read(&cubin).unwrap()).unwrap();
    let f = be.get_function(&module, "plow_spec_accept").unwrap();

    // xorshift; small token alphabet so drafts match often.
    let mut x = 0x2545_f491u32;
    let mut rnd = |m: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % m
    };
    // 11 is also a draft token, so stops land inside accepted runs as well as on corrections.
    let stop = [11u32, 7];
    for round in 0..64 {
        let (mut spans, mut inputs, mut argmax) = (Vec::new(), Vec::new(), Vec::new());
        for s in 0..37u32 {
            let k = (s + round) % (SPEC_KMAX as u32 + 1);
            let row0 = inputs.len() as u32;
            spans.push(SpecSpanDev { row0, k, req_gen: s * 3 + round, _pad: 0 });
            let accept_to = rnd(k + 2);
            for j in 0..=k {
                inputs.push(if j == 0 { 100 + rnd(50) } else { 10 + rnd(4) });
            }
            for j in 0..=k {
                // Agree with the next input up to `accept_to`, then diverge.
                let next = inputs.get((row0 + j + 1) as usize).copied();
                argmax.push(match next {
                    Some(t) if j < accept_to && j < k => t,
                    _ => rnd(12),
                });
            }
        }
        let d = |n: usize| be.alloc(0, (n.max(1) * 4) as u64).unwrap();
        let (d_arg, d_in, d_stop) = (d(argmax.len()), d(inputs.len()), d(stop.len()));
        let d_spans = be.alloc(0, (spans.len() * 16) as u64).unwrap();
        let d_out = be
            .alloc(0, (spans.len() * std::mem::size_of::<SpecVerdict>()) as u64)
            .unwrap();
        be.upload(&d_arg, 0, bytemuck::cast_slice(&argmax)).unwrap();
        be.upload(&d_in, 0, bytemuck::cast_slice(&inputs)).unwrap();
        be.upload(&d_stop, 0, bytemuck::cast_slice(&stop)).unwrap();
        be.upload(&d_spans, 0, bytemuck::cast_slice(&spans)).unwrap();
        let (mut a0, mut a1, mut a2) = (d_arg.base, d_in.base, d_spans.base);
        let (mut a3, mut a4, mut a5, mut a6) =
            (spans.len() as u32, d_stop.base, stop.len() as u32, d_out.base);
        let mut args = [
            &mut a0 as *mut u64 as *mut std::ffi::c_void,
            &mut a1 as *mut u64 as *mut std::ffi::c_void,
            &mut a2 as *mut u64 as *mut std::ffi::c_void,
            &mut a3 as *mut u32 as *mut std::ffi::c_void,
            &mut a4 as *mut u64 as *mut std::ffi::c_void,
            &mut a5 as *mut u32 as *mut std::ffi::c_void,
            &mut a6 as *mut u64 as *mut std::ffi::c_void,
        ];
        be.launch_kernel(f, 1, 64, 0, &mut args, None).unwrap();
        let mut got = vec![SpecVerdict::default(); spans.len()];
        be.download(&d_out, 0, bytemuck::cast_slice_mut(&mut got)).unwrap();
        for (sp, v) in spans.iter().zip(&got) {
            let r = sp.row0 as usize..(sp.row0 + sp.k + 1) as usize;
            let want = accept_greedy(&inputs[r.clone()], &argmax[r], sp.row0, sp.req_gen, &stop);
            assert_eq!(*v, want, "round {round} span {sp:?}");
        }
    }
}
