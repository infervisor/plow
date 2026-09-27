//! `plow_sample_cfg` (device CFG draw) vs the host guided sampler `text::sample::sample_cfg`.
//!
//! Each pair gets its own (cond, uncond) bf16 rows, penalty history and uniform; the device token
//! must equal the host token (float-reduction order may move a draw sitting on a CDF boundary, so
//! the gate is >= 99.5% exact, and greedy is exact). Gated on `PLOW_GPU_TEST=1` (GPU + nvcc;
//! `PLOW_NVCC` or /usr/local/cuda/bin/nvcc).

#![cfg(feature = "cuda")]

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::text::sample::{sample_cfg, CfgParams, SplitMix};

const SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");
const V: usize = 8194;
const PAIRS: usize = 512;

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

#[test]
fn device_cfg_draw_matches_host() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + nvcc)");
        return;
    }
    let dir = std::env::temp_dir().join(format!("plowrt-sample-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, cubin) = (dir.join("s.cu"), dir.join("s.cubin"));
    std::fs::write(&src, SRC).unwrap();
    let nvcc = std::env::var("PLOW_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let out = std::process::Command::new(nvcc)
        .args(["-arch=native", "-O3", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc");
    assert!(out.status.success(), "nvcc: {}", String::from_utf8_lossy(&out.stderr));
    let be = CudaBackend::new(0).expect("CUDA backend");
    let module = be.module_load(&std::fs::read(&cubin).unwrap()).unwrap();
    let f = be.get_function(&module, "plow_sample_cfg").unwrap();
    let threads = be.module_global_u32(&module, "plow_sample_threads").unwrap().unwrap_or(256);
    let stream = be.stream_create().unwrap();

    let b = 2 * PAIRS;
    let mut rng = SplitMix::new(7);
    // Peaked speech-LM-like rows: a few dozen plausible tokens, a long flat tail; the
    // unconditional row is a noisy copy.
    let mut logits = vec![0u16; b * V];
    let mut hist: Vec<Vec<u32>> = Vec::new();
    for p in 0..PAIRS {
        for i in 0..V {
            let r = rng.unit();
            let head = if r < 0.006 { 8.0 + 6.0 * rng.unit() } else { 0.0 };
            let c = head + 2.0 * rng.unit() - 1.0;
            let u = c + 1.5 * (rng.unit() - 0.5);
            logits[2 * p * V + i] = bf16(c);
            logits[(2 * p + 1) * V + i] = bf16(u);
        }
        // History: BOS-like token + a run of generated tokens with repeats.
        let n = (rng.unit() * 200.0) as usize;
        let mut h = vec![6561u32];
        for _ in 0..n {
            h.push((rng.unit() * 64.0) as u32 * 97 % V as u32);
        }
        hist.push(h);
    }
    let f32_of = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let cases = [
        CfgParams { cfg_weight: 0.5, temperature: 0.8, min_p: 0.05, top_p: 1.0, repetition_penalty: 1.2 },
        CfgParams { cfg_weight: 0.3, temperature: 1.0, min_p: 0.0, top_p: 1.0, repetition_penalty: 1.0 },
        CfgParams { cfg_weight: 0.5, temperature: 0.8, min_p: 0.02, top_p: 0.9, repetition_penalty: 1.3 },
    ];

    let d_logits = be.alloc(0, (b * V * 2) as u64).unwrap();
    be.upload(&d_logits, 0, bytemuck::cast_slice(&logits)).unwrap();
    let d_ids = be.alloc(0, (b * 4) as u64).unwrap();
    let d_prm = be.alloc(0, (6 * b * 4) as u64).unwrap();
    let d_rng = be.alloc(0, (b * 4) as u64).unwrap();
    let d_cnt = be.alloc(0, (b * V * 4) as u64).unwrap();
    let d_es = be.alloc(0, (b * V * 4) as u64).unwrap();

    for (ci, p) in cases.iter().enumerate() {
        for greedy in [false, true] {
            let mut counts = vec![0u32; b * V];
            for (q, h) in hist.iter().enumerate() {
                for &t in h {
                    counts[2 * q * V + t as usize] += 1;
                }
            }
            be.upload(&d_cnt, 0, bytemuck::cast_slice(&counts)).unwrap();
            let mut prm = vec![0f32; 6 * b];
            let mut us = vec![0f32; b];
            for q in 0..PAIRS {
                let o = 2 * q;
                for (j, v) in [1.0, p.cfg_weight, p.repetition_penalty, if greedy { 0.0 } else { p.temperature }, p.top_p, p.min_p]
                    .into_iter()
                    .enumerate()
                {
                    prm[j * b + o] = v;
                }
                us[o] = (q as f32 + 0.5) / PAIRS as f32;
            }
            be.upload(&d_prm, 0, bytemuck::cast_slice(&prm)).unwrap();
            be.upload(&d_rng, 0, bytemuck::cast_slice(&us)).unwrap();
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) =
                (d_logits.base, d_ids.base, d_prm.base, d_rng.base, d_cnt.base, d_es.base);
            let (mut av, mut ab) = (V as u32, b as u32);
            let mut args = [
                &mut a0 as *mut u64 as *mut std::ffi::c_void,
                &mut a1 as *mut u64 as *mut std::ffi::c_void,
                &mut a2 as *mut u64 as *mut std::ffi::c_void,
                &mut a3 as *mut u64 as *mut std::ffi::c_void,
                &mut a4 as *mut u64 as *mut std::ffi::c_void,
                &mut a5 as *mut u64 as *mut std::ffi::c_void,
                &mut av as *mut u32 as *mut std::ffi::c_void,
                &mut ab as *mut u32 as *mut std::ffi::c_void,
            ];
            be.launch_kernel(f, b as u32, threads, 0, &mut args, Some(&stream)).unwrap();
            be.stream_synchronize(&stream).unwrap();
            let mut ids = vec![0i32; b];
            be.download(&d_ids, 0, bytemuck::cast_slice_mut(&mut ids)).unwrap();
            let mut cnt_after = vec![0u32; b * V];
            be.download(&d_cnt, 0, bytemuck::cast_slice_mut(&mut cnt_after)).unwrap();

            let mut same = 0;
            let mut scratch = Vec::new();
            for q in 0..PAIRS {
                let o = 2 * q;
                let cond: Vec<f32> = logits[o * V..(o + 1) * V].iter().map(|&x| f32_of(x)).collect();
                let unc: Vec<f32> = logits[(o + 1) * V..(o + 2) * V].iter().map(|&x| f32_of(x)).collect();
                let u = (!greedy).then_some(us[o]);
                let host = sample_cfg(p, &cond, &unc, hist[q].iter().copied(), u, &mut scratch);
                assert_eq!(ids[o], ids[o + 1], "pair {q}: both members get the token");
                assert_eq!(cnt_after[o * V + ids[o] as usize], counts[o * V + ids[o] as usize] + 1);
                same += (ids[o] as u32 == host) as usize;
            }
            eprintln!("case {ci} greedy={greedy}: {same}/{PAIRS} device == host");
            if greedy {
                assert_eq!(same, PAIRS);
            } else {
                assert!(same * 1000 >= PAIRS * 995, "case {ci}: {same}/{PAIRS}");
            }
        }
    }
}
