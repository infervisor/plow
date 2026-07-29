//! Hardware test for the HSA engine primitives (`device::hsa`).
//!
//! These are the surface `exec::gpu` needs beyond the `Backend` trait — pinned
//! host staging, the async copy pair, device→device copy, and the event/stream
//! ordering handles. They are the AMD counterparts of the `CudaBackend` methods
//! the engine was written against, so they are worth testing on their own
//! BEFORE the engine is made generic over them: a bug here would otherwise
//! surface as a wrong token deep inside a serving step.
//!
//! Needs a real gfx9xx GPU + ROCr. Gated on `PLOW_GPU_TEST=1` like the other
//! device tests, so a CI box without a GPU stays green.
//!
//!   PLOW_GPU_TEST=1 cargo test -p plowrt --features hsa --test hsa_primitives

#![cfg(feature = "hsa")]

use plowrt::device::hsa::HsaBackend;
use plowrt::device::Backend;

fn gpu_enabled() -> bool {
    std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1")
}

/// PLOW_GPU_TEST=1 is an assertion that a GPU is present, so a failed probe is
/// a FAILURE, not a skip. Returning `None` here once turned a missing
/// `libelf.so.1` (ROCr never loaded, CPU fallback taken) into five green
/// "ok"s — the run reported success while testing nothing at all.
///
/// ONE backend for the whole process, not one per test. HSA is a process-global
/// runtime: a second `hsa_init` after a backend has been built fails with 4104
/// (OUT_OF_RESOURCES), so per-test construction passed the first test and
/// failed every one after it. plowrt itself holds a single backend per device,
/// so sharing here also matches how the code is really used.
fn backend() -> &'static HsaBackend {
    static BE: std::sync::OnceLock<HsaBackend> = std::sync::OnceLock::new();
    BE.get_or_init(|| {
        HsaBackend::new(0).unwrap_or_else(|e| {
            panic!(
                "PLOW_GPU_TEST=1 but no HSA device: {e}\n\
                 If this is a dlopen failure, ROCr's deps are missing from \
                 LD_LIBRARY_PATH — run inside `nix develop`, which wires them."
            )
        })
    })
}

/// REGRESSION GUARD. This hung forever until `HSA_SIGNAL_CONDITION_LT` was
/// corrected from 0 (which is EQ) to 2: the completion wait meant "block until
/// the signal equals 1" while the signal counts down to 0. If this test hangs
/// again, check those constants before anything else.
#[test]
fn pinned_staging_roundtrips_through_device() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();

    // The engine's staging slab is written by the host and read by the very
    // next packet, so the test mirrors that: fill pinned host memory, push it
    // to the device, pull it back into a second pinned slab, compare.
    const N: usize = 64 * 1024;

    // The staging slab must behave as ordinary host memory for the fill.
    let mut slab = be.host_alloc_pinned(N).expect("host_alloc_pinned");
    for (i, b) in slab.as_mut_slice().iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let expect: Vec<u8> = slab.as_slice().to_vec();

    // The copy SOURCE is pageable host memory, which is what
    // `memcpy_htod_async` promises to take: it pins the source with
    // `hsa_amd_memory_lock`, and locking an already-device-accessible
    // fine-grained POOL allocation is invalid (returns HSA_STATUS_ERROR 4096 —
    // measured). Copying straight out of a `HsaPinned` slab therefore needs a
    // separate direct-async_copy entry point, which the engine will want and
    // this test deliberately does not pretend exists yet.
    let dev = be.alloc(0, N as u64).expect("alloc");
    let stream = be.stream_create().expect("stream_create");

    // SAFETY: `expect` outlives the copy — synchronised before it is read back.
    unsafe { be.memcpy_htod_async(dev.base, &expect, &stream) }.expect("htod");
    be.stream_synchronize(&stream).expect("sync");

    let mut back = vec![0u8; N];
    // SAFETY: `back` outlives the copy.
    unsafe { be.memcpy_dtoh_async(&mut back, dev.base, &stream) }.expect("dtoh");
    be.stream_synchronize(&stream).expect("sync");

    assert_eq!(back, expect, "H2D→D2H roundtrip corrupted bytes");
}

#[test]
fn device_to_device_copy_is_exact() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();

    // dtod backs the KV-cache row moves; an off-by-one here would corrupt
    // attention silently rather than fail loudly.
    const N: usize = 8192;
    let pattern: Vec<u8> = (0..N).map(|i| (i * 7 % 253) as u8).collect();

    let a = be.alloc(0, N as u64).expect("alloc a");
    let b = be.alloc(0, N as u64).expect("alloc b");
    be.memcpy_htod(a.base, &pattern).expect("htod");
    be.memcpy_dtod(b.base, a.base, N as u64).expect("dtod");

    let mut back = vec![0u8; N];
    be.download(&b, 0, &mut back).expect("download");
    assert_eq!(back, pattern, "dtod copy did not reproduce the source bytes");
}

#[test]
fn events_order_and_report_elapsed() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();

    let stream = be.stream_create().expect("stream_create");
    let start = be.event_create(true).expect("event_create");
    let end = be.event_create(true).expect("event_create");

    // Deliberately NOT a device copy: that path is the open hang documented on
    // `pinned_staging_roundtrips_through_device`. What is under test here is
    // the event clock itself — that a timing event stamps at record, that the
    // stamp survives a synchronize, and that the delta is a real interval.
    const SLEEP_MS: u64 = 20;
    be.event_record(&start, &stream).expect("record start");
    be.event_synchronize(&start).expect("sync start");
    std::thread::sleep(std::time::Duration::from_millis(SLEEP_MS));
    be.event_record(&end, &stream).expect("record end");
    be.event_synchronize(&end).expect("sync end");

    let ms = be.event_elapsed_ms(&start, &end).expect("elapsed");
    assert!(
        ms >= SLEEP_MS as f32 * 0.5,
        "elapsed {ms} ms did not span a {SLEEP_MS} ms sleep — is the stamp taken at record?"
    );
    assert!(ms < 60_000.0, "implausible elapsed: {ms} ms");
}

#[test]
fn sync_only_events_report_zero_elapsed() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();

    // `event_create(false)` is the cheap buffer-reuse gate in `UploadPipe`: it
    // must still order correctly, but it carries no clock, and reading elapsed
    // off it must yield 0.0 rather than failing a serving step.
    let stream = be.stream_create().expect("stream_create");
    let a = be.event_create(false).expect("event_create");
    let b = be.event_create(false).expect("event_create");
    be.event_record(&a, &stream).expect("record a");
    be.event_synchronize(&a).expect("sync a");
    be.event_record(&b, &stream).expect("record b");
    be.event_synchronize(&b).expect("sync b");
    assert_eq!(be.event_elapsed_ms(&a, &b).expect("elapsed"), 0.0);
}

#[test]
fn executor_geometry_matches_the_agent() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();

    // The engine sizes its cooperative grid from these three, so a zero here
    // becomes a zero-block launch. The UPPER bound matters just as much: with
    // the agent-info enum off by two, `cu_count` held the PCI chip id (30115)
    // and a `> 0` assertion waved it through. No shipping GPU has >1024 CUs.
    assert!(
        (1..=1024).contains(&be.sm_count()),
        "implausible CU count {} — check HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT",
        be.sm_count()
    );
    assert!(be.lds_bytes() > 0, "lds budget must be positive");
    assert!(
        be.wave_width() == 64 || be.wave_width() == 32,
        "unexpected wave width {}",
        be.wave_width()
    );
    eprintln!(
        "agent: {} CUs={} lds={}B wave={}",
        be.device_name(),
        be.sm_count(),
        be.lds_bytes(),
        be.wave_width()
    );
}
