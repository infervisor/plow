//! `plow_spec_accept` (runtime/nvidia/sample_sm120.cu) against the host golden
//! `orch::speculative::accept_greedy` on random spans: every k in 0..=15, partial and full
//! acceptance, stop tokens at every emission position. Sampled spans: `plow_sample` over the verify
//! rows then `plow_spec_accept` against the host reference `accept_sampled` + `host_row_sample`
//! for the same draws.
//!
//! Gated on `PLOW_GPU_TEST=1` (needs GPU + nvcc). Skips silently otherwise.

#![cfg(feature = "cuda")]

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::orch::speculative::{
    accept_greedy, accept_sampled, host_row_sample, SpecSpanDev, SpecVerdict, SPEC_KMAX,
};

const SRC: &str = include_str!("../../../runtime/nvidia/sample_sm120.cu");

fn gpu_test() -> bool {
    let on = std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1");
    if !on {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + nvcc)");
    }
    on
}

fn build_cubin(tag: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("plowrt-spec-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("sample_sm120.cu");
    let cubin = dir.join("sample_sm120.cubin");
    std::fs::write(&src, SRC).unwrap();
    let nvcc = std::env::var("PLOW_NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let mut cmd = std::process::Command::new(&nvcc);
    cmd.env_clear()
        .env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin");
    if let Ok(flags) = std::env::var("NVCC_PREPEND_FLAGS") {
        cmd.env("NVCC_PREPEND_FLAGS", flags);
    }
    let out = cmd
        .args(["-arch=native", "-O3", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc not found");
    assert!(
        out.status.success(),
        "nvcc: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(&cubin).unwrap()
}

#[test]
fn device_accept_matches_host_golden() {
    if !gpu_test() {
        return;
    }
    let be = CudaBackend::new(0).expect("CUDA backend");
    let module = be.module_load(&build_cubin("accept")).unwrap();
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
            spans.push(SpecSpanDev {
                row0,
                k,
                req_gen: s * 3 + round,
                _pad: 0,
            });
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
        be.upload(&d_spans, 0, bytemuck::cast_slice(&spans))
            .unwrap();
        let (mut a0, mut a1, mut a2) = (d_arg.base, d_in.base, d_spans.base);
        let (mut a3, mut a4, mut a5, mut a6) = (
            spans.len() as u32,
            d_stop.base,
            stop.len() as u32,
            d_out.base,
        );
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
        be.download(&d_out, 0, bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for (sp, v) in spans.iter().zip(&got) {
            let r = sp.row0 as usize..(sp.row0 + sp.k + 1) as usize;
            let want = accept_greedy(&inputs[r.clone()], &argmax[r], sp.row0, sp.req_gen, &stop);
            assert_eq!(*v, want, "round {round} span {sp:?}");
        }
    }
}

fn args(a: &mut [u64]) -> Vec<*mut std::ffi::c_void> {
    a.iter_mut()
        .map(|x| x as *mut u64 as *mut std::ffi::c_void)
        .collect()
}

/// Sampled spans with truncating parameters (the `sample_select` path plain device decode
/// takes): the device verdict equals the host reference for the same per-row draws, row by row
/// through the first mismatch and the bonus row.
#[test]
fn device_sampled_accept_matches_host_reference() {
    if !gpu_test() {
        return;
    }
    let be = CudaBackend::new(0).expect("CUDA backend");
    let module = be.module_load(&build_cubin("sampled")).unwrap();
    let f_sample = be.get_function(&module, "plow_sample").unwrap();
    let f_accept = be.get_function(&module, "plow_spec_accept").unwrap();
    let threads = be
        .module_global_u32(&module, "plow_sample_threads")
        .unwrap()
        .unwrap_or(256);
    const V: usize = 8192;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let (mut spans_total, mut rows_total, mut accepted) = (0usize, 0usize, 0usize);
    let mut prior = Vec::new();
    for (temp, top_k, top_p, min_p) in [
        (0.7f32, 0usize, 0.95f32, 0.0f32),
        (1.0, 40, 1.0, 0.0),
        (0.8, 0, 1.0, 0.05),
        (0.7, 10, 0.9, 0.02),
        (1.3, 64, 0.9, 0.0),
    ] {
        let params = plowrt::text::sample::SamplingParams {
            temperature: temp,
            top_k,
            top_p,
            min_p,
            ..Default::default()
        };
        for round in 0..8u32 {
            let (mut spans, mut inputs, mut bits, mut host_rows, mut draws) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for s in 0..6u32 {
                let k = (s + round) % 8;
                let row0 = inputs.len() as u32;
                spans.push(SpecSpanDev {
                    row0,
                    k,
                    req_gen: s,
                    _pad: 0,
                });
                inputs.push(V as u32 - 1 - s);
                for j in 0..=k {
                    // A peaked row: a handful of strong tokens over a noisy floor, rounded to bf16.
                    let row: Vec<f32> = (0..V)
                        .map(|i| {
                            let l = 3.0 * (unit() + unit() + unit() - 1.5);
                            let spike = if i % 997 == (j as usize * 31 + s as usize) % 997 {
                                6.0 + 4.0 * unit()
                            } else {
                                0.0
                            };
                            f32::from_bits((l + spike).to_bits() & 0xFFFF_0000)
                        })
                        .collect();
                    bits.extend(row.iter().map(|v| (v.to_bits() >> 16) as u16));
                    draws.push((unit() * 10_000.0).floor() / 10_000.0);
                    if j < k {
                        // The draft: this row's own sample under an independent draw.
                        let mut l = row.clone();
                        inputs.push(host_row_sample(
                            &mut l,
                            &params,
                            &[],
                            &[],
                            unit(),
                            &mut prior,
                        ));
                    }
                    host_rows.push(row);
                }
            }
            let rows = inputs.len();
            let d = |n: usize| be.alloc(0, (n.max(1) * 4) as u64).unwrap();
            let d_logits = be.alloc(0, (rows * V * 2) as u64).unwrap();
            let d_scratch = be.alloc(0, (rows * V * 4) as u64).unwrap();
            let (d_ids, d_in, d_t, d_k, d_p, d_m, d_u) = (
                d(rows),
                d(rows),
                d(rows),
                d(rows),
                d(rows),
                d(rows),
                d(rows),
            );
            let d_spans = be.alloc(0, (spans.len() * 16) as u64).unwrap();
            let d_out = be
                .alloc(0, (spans.len() * std::mem::size_of::<SpecVerdict>()) as u64)
                .unwrap();
            be.upload(&d_logits, 0, bytemuck::cast_slice(&bits))
                .unwrap();
            be.upload(&d_in, 0, bytemuck::cast_slice(&inputs)).unwrap();
            be.upload(&d_t, 0, bytemuck::cast_slice(&vec![temp; rows]))
                .unwrap();
            be.upload(&d_k, 0, bytemuck::cast_slice(&vec![top_k as i32; rows]))
                .unwrap();
            be.upload(&d_p, 0, bytemuck::cast_slice(&vec![top_p; rows]))
                .unwrap();
            be.upload(&d_m, 0, bytemuck::cast_slice(&vec![min_p; rows]))
                .unwrap();
            be.upload(&d_u, 0, bytemuck::cast_slice(&draws)).unwrap();
            be.upload(&d_spans, 0, bytemuck::cast_slice(&spans))
                .unwrap();
            let mut a = [
                d_logits.base,
                d_ids.base,
                d_t.base,
                d_k.base,
                d_p.base,
                d_m.base,
                d_u.base,
                d_scratch.base,
            ];
            let (mut v32, mut b32) = (V as u32, rows as u32);
            let mut ptrs = args(&mut a);
            ptrs.push(&mut v32 as *mut u32 as *mut std::ffi::c_void);
            ptrs.push(&mut b32 as *mut u32 as *mut std::ffi::c_void);
            be.launch_kernel(f_sample, rows as u32, threads, 0, &mut ptrs, None)
                .unwrap();
            let mut b = [d_ids.base, d_in.base, d_spans.base, 0, 0, 0, d_out.base];
            let (mut n_spans, mut n_stop) = (spans.len() as u32, 0u32);
            let mut ptrs = args(&mut b);
            ptrs[3] = &mut n_spans as *mut u32 as *mut std::ffi::c_void;
            ptrs[5] = &mut n_stop as *mut u32 as *mut std::ffi::c_void;
            be.launch_kernel(f_accept, 1, 64, 0, &mut ptrs, None)
                .unwrap();
            let mut got = vec![SpecVerdict::default(); spans.len()];
            be.download(&d_out, 0, bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for (sp, v) in spans.iter().zip(&got) {
                let r0 = sp.row0 as usize;
                let want = accept_sampled::<()>(
                    &inputs[r0..r0 + sp.k as usize + 1],
                    |j| {
                        rows_total += 1;
                        let mut l = host_rows[r0 + j].clone();
                        Ok(host_row_sample(
                            &mut l,
                            &params,
                            &[],
                            &[],
                            draws[r0 + j],
                            &mut prior,
                        ))
                    },
                    sp.row0,
                    sp.req_gen,
                    &[],
                )
                .unwrap();
                assert_eq!(
                    *v, want,
                    "T{temp} k{top_k} p{top_p} m{min_p} round {round} span {sp:?}"
                );
                spans_total += 1;
                accepted += usize::from(v.n_acc);
            }
        }
    }
    eprintln!("sampled spans {spans_total}, host rows {rows_total}, drafts accepted {accepted}");
    assert!(accepted > 0 && accepted < rows_total);
}
