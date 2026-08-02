//! CUDA backend smoke test against the real driver (feature `cuda`).
//!
//! Driver-API only — no cudart. Compiles a trivial vector-add cubin with nvcc
//! at test time, then exercises the full backend surface the engine relies on:
//! alloc → upload → module_load → get_function → launch_cooperative → sync →
//! download.
//!
//! Gated on `PLOW_GPU_TEST=1` (needs a GPU + nvcc; run under
//! `gpulease serve-be cargo test ...`). Skips silently otherwise so
//! `cargo test -p plowrt --features cuda` stays green on a driverless box.

#![cfg(feature = "cuda")]

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;

const VADD_CU: &str = r#"
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

#[test]
fn vector_add_end_to_end() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + nvcc)");
        return;
    }

    // Build the cubin.
    let dir = std::env::temp_dir().join(format!("plowrt-vadd-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("vadd.cu");
    let cubin = dir.join("vadd.cubin");
    std::fs::write(&src, VADD_CU).unwrap();
    // env_clear: under `nix develop`, CPATH/NIX_CFLAGS point nvcc's host pass
    // at nix glibc headers that conflict with the CUDA math headers.
    let out = std::process::Command::new("/usr/local/cuda/bin/nvcc")
        .env_clear()
        .env("PATH", "/usr/local/cuda/bin:/usr/bin:/bin")
        .args(["-arch=native", "-cubin", "-o"])
        .arg(&cubin)
        .arg(&src)
        .output()
        .expect("nvcc not found");
    assert!(
        out.status.success(),
        "nvcc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let image = std::fs::read(&cubin).unwrap();

    let be = CudaBackend::new(0).expect("CUDA backend bringup");
    let targets = be.enumerate();
    assert!(!targets.is_empty(), "no executors enumerated");
    assert_eq!(targets.len() as u32, be.sm_count());
    eprintln!("device: {} ({} SMs)", be.device_name(), be.sm_count());

    const N: usize = 4096;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

    let d_a = be.alloc(0, (N * 4) as u64).unwrap();
    let d_b = be.alloc(0, (N * 4) as u64).unwrap();
    let d_c = be.alloc(0, (N * 4) as u64).unwrap();
    be.upload(&d_a, 0, bytemuck::cast_slice(&a)).unwrap();
    be.upload(&d_b, 0, bytemuck::cast_slice(&b)).unwrap();

    let module = be.module_load(&image).unwrap();
    let f = be.get_function(&module, "vadd").unwrap();
    // 4096 elements / 256 threads = 16 blocks — trivially co-resident.
    let occ = be.occupancy_blocks_per_sm(f, 256, 0).unwrap();
    assert!(occ >= 1, "vadd kernel reports zero occupancy");

    let mut pa = d_a.base;
    let mut pb = d_b.base;
    let mut pc = d_c.base;
    let mut n = N as i32;
    let mut params = [
        &mut pa as *mut _ as *mut std::ffi::c_void,
        &mut pb as *mut _ as *mut std::ffi::c_void,
        &mut pc as *mut _ as *mut std::ffi::c_void,
        &mut n as *mut _ as *mut std::ffi::c_void,
    ];
    be.launch_cooperative(f, (N as u32).div_ceil(256), 256, 0, &mut params, None)
        .unwrap();
    be.synchronize().unwrap();

    let mut c = vec![0f32; N];
    be.download(&d_c, 0, bytemuck::cast_slice_mut(&mut c))
        .unwrap();
    for i in 0..N {
        assert_eq!(c[i], (3 * i) as f32, "c[{i}]");
    }

    // The out-of-range guards must reject, not corrupt.
    assert!(be.upload(&d_a, (N * 4) as u64, &[0u8; 4]).is_err());
    let mut one = [0u8; 8];
    assert!(be.download(&d_a, (N * 4 - 4) as u64, &mut one).is_err());

    // ---- async submission surface (plan stage 1) ----
    // The engine's steady-state step shape: pinned staging → async H2D →
    // async memset → cooperative launch on the stream → async D2H → ONE
    // stream synchronize. Events bracket the queue; no cuCtxSynchronize.
    let stream = be.stream_create().unwrap();
    let ev_start = be.event_create(true).unwrap();
    let ev_end = be.event_create(true).unwrap();

    let mut pin = be.host_alloc_pinned(N * 4).unwrap();
    bytemuck::cast_slice_mut::<u8, f32>(pin.as_mut_slice())
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = (10 * i) as f32);

    be.event_record(&ev_start, &stream).unwrap();
    // SAFETY: `pin` and the device buffers outlive the stream synchronize.
    unsafe {
        be.memcpy_htod_async(d_a.base, pin.as_slice(), &stream)
            .unwrap();
    }
    be.memset_d8_async(d_b.base, 0, N * 4, &stream).unwrap();
    let mut params = [
        &mut pa as *mut _ as *mut std::ffi::c_void,
        &mut pb as *mut _ as *mut std::ffi::c_void,
        &mut pc as *mut _ as *mut std::ffi::c_void,
        &mut n as *mut _ as *mut std::ffi::c_void,
    ];
    be.launch_cooperative(
        f,
        (N as u32).div_ceil(256),
        256,
        0,
        &mut params,
        Some(&stream),
    )
    .unwrap();
    // SAFETY: as above — the readback retires before the synchronize returns.
    unsafe {
        be.memcpy_dtoh_async(pin.as_mut_slice(), d_c.base, &stream)
            .unwrap();
    }
    be.event_record(&ev_end, &stream).unwrap();
    be.stream_synchronize(&stream).unwrap();

    // a = 10i (pinned upload), b = 0 (async memset) → c = 10i.
    let c: &[f32] = bytemuck::cast_slice(pin.as_slice());
    for (i, &v) in c.iter().enumerate() {
        assert_eq!(v, (10 * i) as f32, "async c[{i}]");
    }
    // Events retired with the stream; the device-timeline delta is sane.
    assert!(
        be.event_query(&ev_end).unwrap(),
        "event not retired after sync"
    );
    be.event_synchronize(&ev_end).unwrap();
    let ms = be.event_elapsed_ms(&ev_start, &ev_end).unwrap();
    assert!(
        (0.0..10_000.0).contains(&ms),
        "elapsed {ms} ms out of range"
    );
    eprintln!("async round-trip device time: {ms:.3} ms");
}
