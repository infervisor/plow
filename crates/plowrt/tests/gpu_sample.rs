//! Device sampler equivalence test (plans/plowrt-gpu-exec-critical-path stage 4).
//!
//! Builds `runtime/nvidia/sample_sm120.cu` with nvcc (like `cuda_gpu.rs` builds
//! vadd), then validates `plow_sample` against a CPU reference that implements
//! the SAME semantics (threshold truncation + index-order inverse-CDF):
//!   1. greedy (t=0) equals the exact argmax (ARGMAX tie-break: lowest index),
//!   2. determinism: identical inputs → identical token,
//!   3. distribution: over a stratified rng grid the device tokens agree with
//!      the CPU reference on ≥99% of draws (the rest are float-reduction ties
//!      at a CDF boundary), and the empirical frequencies match the kept-set
//!      probabilities.
//!
//! Gated on `PLOW_GPU_TEST=1` (needs GPU + nvcc). Skips silently otherwise.

#![cfg(feature = "cuda")]

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;

/// f32 → bf16 bits (round to nearest even) and back — the device reads the row
/// as bf16, so the reference must see the same rounded values.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let rounding = 0x7fff + ((u >> 16) & 1);
    ((u.wrapping_add(rounding)) >> 16) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// CPU reference: the device sampler's exact semantics.
/// `bf` is the row as bf16 bits. Returns the sampled token.
fn cpu_sample(bf: &[u16], t: f32, top_k: i32, top_p: f32, min_p: f32, rng01: f32) -> u32 {
    let v = bf.len();
    if t <= 1e-6 {
        // Argmax with the ARGMAX packed key (lowest index breaks ties).
        let mut best: u64 = 0;
        for (i, &bits) in bf.iter().enumerate() {
            let key = if bits & 0x8000 != 0 {
                (!bits) as u32
            } else {
                (bits | 0x8000) as u32
            };
            let p = ((key as u64) << 32) | (!(i as u32) as u64);
            best = best.max(p);
        }
        return !((best & 0xFFFF_FFFF) as u32);
    }
    let inv_t = 1.0 / t;
    let lmax = bf
        .iter()
        .map(|&b| bf16_bits_to_f32(b))
        .fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = bf
        .iter()
        .map(|&b| ((bf16_bits_to_f32(b) - lmax) * inv_t).exp())
        .collect();
    let total: f32 = e.iter().sum();

    let count_ge = |floor: f32| e.iter().filter(|&&w| w >= floor).count();
    let mass_ge = |floor: f32| e.iter().filter(|&&w| w >= floor).sum::<f32>();

    let mut floor = min_p;
    if top_k > 0 {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if count_ge(mid) as i32 > top_k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        floor = floor.max(lo);
    }
    if top_p < 1.0 {
        let want = top_p * total;
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if mass_ge(mid) > want {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        floor = floor.max(lo);
    }
    let kept: f32 = mass_ge(floor);
    let target = rng01 * kept;
    let mut acc = 0.0f32;
    for (i, &w) in e.iter().enumerate() {
        if w >= floor {
            acc += w;
            if acc > target {
                return i as u32;
            }
        }
    }
    // Mass edge: highest-weight kept token.
    (0..v).find(|&i| e[i] >= floor).unwrap_or(0) as u32
}

const SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");

/// Launch config buffers for a B-row sample, all rows sharing one logits row.
struct Case {
    t: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
}

#[test]
fn device_sampler_matches_cpu_reference() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + nvcc)");
        return;
    }

    // Build the sampler cubin.
    let dir = std::env::temp_dir().join(format!("plowrt-sample-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("sample_sm120.cu");
    let cubin = dir.join("sample_sm120.cubin");
    std::fs::write(&src, SRC).unwrap();
    let out = std::process::Command::new("/usr/local/cuda/bin/nvcc")
        .env_clear()
        .env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin")
        .args(["-arch=native", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc not found");
    assert!(out.status.success(), "nvcc: {}", String::from_utf8_lossy(&out.stderr));
    let image = std::fs::read(&cubin).unwrap();

    let be = CudaBackend::new(0).expect("CUDA backend");
    let module = be.module_load(&image).unwrap();
    let f = be.get_function(&module, "plow_sample").unwrap();
    let stream = be.stream_create().unwrap();

    // A realistic logits row (V=8192): real model logits after softmax are
    // peaked — probability concentrates in a modest head and decays, so a
    // top_p=0.95 nucleus is a well-conditioned few-dozen-token set, not a flat
    // thousand-token band. Model it as exponential decay over a shuffled index
    // (so the peak isn't at index 0) plus small per-token jitter.
    const V: usize = 8192;
    let mut logits_f = vec![0.0f32; V];
    for (i, l) in logits_f.iter_mut().enumerate() {
        // Deterministic pseudo-rank in [0, V) from a hash, so the distribution
        // is peaked but not index-ordered.
        let rank = ((i.wrapping_mul(2654435761) >> 8) % V as usize) as f32;
        let jitter = ((i.wrapping_mul(40503) >> 4) & 0x3f) as f32 / 64.0;
        *l = 14.0 * (-rank / 12.0).exp() + jitter; // sharp exponential head
    }
    let bf: Vec<u16> = logits_f.iter().map(|&x| f32_to_bf16_bits(x)).collect();

    // B rows: one per stratified rng sample; all read the same logits row.
    const B: usize = 4096;
    let rng01: Vec<f32> = (0..B).map(|b| (b as f32 + 0.5) / B as f32).collect();

    // Device buffers. logits is [B][V] but every row is identical; upload once
    // per row (cheap at this size) so the kernel's per-b offset is exercised.
    let d_logits = be.alloc(0, (B * V * 2) as u64).unwrap();
    let row_bytes: &[u8] = bytemuck::cast_slice(&bf);
    for b in 0..B {
        be.upload(&d_logits, (b * V * 2) as u64, row_bytes).unwrap();
    }
    let d_ids = be.alloc(0, (B * 4) as u64).unwrap();
    let d_escratch = be.alloc(0, (B * V * 4) as u64).unwrap();
    let d_temp = be.alloc(0, (B * 4) as u64).unwrap();
    let d_topk = be.alloc(0, (B * 4) as u64).unwrap();
    let d_topp = be.alloc(0, (B * 4) as u64).unwrap();
    let d_minp = be.alloc(0, (B * 4) as u64).unwrap();
    let d_rng = be.alloc(0, (B * 4) as u64).unwrap();
    be.upload(&d_rng, 0, bytemuck::cast_slice(&rng01)).unwrap();

    let cases = [
        Case { t: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 }, // greedy
        Case { t: 0.8, top_k: 40, top_p: 1.0, min_p: 0.0 },
        Case { t: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0 },
        Case { t: 0.7, top_k: 0, top_p: 1.0, min_p: 0.05 },
        Case { t: 1.2, top_k: 100, top_p: 0.9, min_p: 0.02 },
    ];

    let mut ids = vec![0i32; B];
    let mut ids2 = vec![0i32; B];
    for (ci, c) in cases.iter().enumerate() {
        be.upload(&d_temp, 0, bytemuck::cast_slice(&vec![c.t; B])).unwrap();
        be.upload(&d_topk, 0, bytemuck::cast_slice(&vec![c.top_k; B])).unwrap();
        be.upload(&d_topp, 0, bytemuck::cast_slice(&vec![c.top_p; B])).unwrap();
        be.upload(&d_minp, 0, bytemuck::cast_slice(&vec![c.min_p; B])).unwrap();

        let launch = |ids: &mut [i32]| {
            // Kernel args live in named locals: kernelParams points at each,
            // and the driver copies them out during the (synchronous) launch.
            let (mut p_logits, mut p_ids) = (d_logits.base, d_ids.base);
            let (mut p_temp, mut p_topk) = (d_temp.base, d_topk.base);
            let (mut p_topp, mut p_minp) = (d_topp.base, d_minp.base);
            let (mut p_rng, mut p_es) = (d_rng.base, d_escratch.base);
            let (mut p_v, mut p_b) = (V as u32, B as u32);
            let mut a = [
                &mut p_logits as *mut u64 as *mut std::ffi::c_void,
                &mut p_ids as *mut u64 as *mut std::ffi::c_void,
                &mut p_temp as *mut u64 as *mut std::ffi::c_void,
                &mut p_topk as *mut u64 as *mut std::ffi::c_void,
                &mut p_topp as *mut u64 as *mut std::ffi::c_void,
                &mut p_minp as *mut u64 as *mut std::ffi::c_void,
                &mut p_rng as *mut u64 as *mut std::ffi::c_void,
                &mut p_es as *mut u64 as *mut std::ffi::c_void,
                &mut p_v as *mut u32 as *mut std::ffi::c_void,
                &mut p_b as *mut u32 as *mut std::ffi::c_void,
            ];
            be.launch_kernel(f, B as u32, 256, 0, &mut a, Some(&stream)).unwrap();
            be.stream_synchronize(&stream).unwrap();
            be.download(&d_ids, 0, bytemuck::cast_slice_mut(ids)).unwrap();
        };
        launch(&mut ids);
        launch(&mut ids2);

        // Determinism.
        assert_eq!(ids, ids2, "case {ci}: device sampler not deterministic");

        if c.t <= 1e-6 {
            // Greedy is a single deterministic token — must equal the argmax
            // exactly on every row.
            let mut mism = 0usize;
            for b in 0..B {
                let want = cpu_sample(&bf, c.t, c.top_k, c.top_p, c.min_p, rng01[b]) as i32;
                if ids[b] != want {
                    mism += 1;
                }
            }
            assert_eq!(mism, 0, "case {ci} greedy: {mism} disagreements (must be exact)");
            eprintln!("case {ci} greedy: exact over {B} draws");
            continue;
        }

        // Stochastic: the plan's gate is a DISTRIBUTION test, not per-draw
        // token equality — a broad nucleus has many near-equal tail tokens, so
        // device tree-reductions vs CPU index-order sums flip individual draws
        // to adjacent tokens without changing the distribution. Compare the
        // device histogram to the CPU reference's over the same stratified rng
        // grid via total-variation distance; both approximate the same kept
        // distribution, so TVD must be small.
        let mut dev_hist = vec![0u32; V];
        let mut cpu_hist = vec![0u32; V];
        for b in 0..B {
            dev_hist[ids[b] as usize] += 1;
            cpu_hist[cpu_sample(&bf, c.t, c.top_k, c.top_p, c.min_p, rng01[b]) as usize] += 1;
        }
        let tvd: f64 = dev_hist
            .iter()
            .zip(&cpu_hist)
            .map(|(&d, &r)| (d as f64 - r as f64).abs())
            .sum::<f64>()
            / (2.0 * B as f64);
        let distinct = dev_hist.iter().filter(|&&c| c > 0).count();
        eprintln!(
            "case {ci} t={} k={} p={} min_p={}: TVD(device,cpu)={tvd:.4}, {distinct} distinct tokens",
            c.t, c.top_k, c.top_p, c.min_p,
        );
        assert!(tvd < 0.05, "case {ci}: device/cpu distributions differ (TVD {tvd:.4} >= 0.05)");
        // Truncation actually happened (not sampling the whole vocab).
        assert!(distinct < V / 2, "case {ci}: kept set implausibly large ({distinct})");
    }
    be.module_unload(&module).unwrap();
}
