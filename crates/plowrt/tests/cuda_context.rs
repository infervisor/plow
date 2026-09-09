#![cfg(feature = "cuda")]
use plowrt::device::{cuda::CudaBackend, Backend};

#[test]
fn opening_another_gpu_does_not_redirect_existing_backend_allocations() {
    let Some(_test) = setup() else { return };
    let a = CudaBackend::new(0).unwrap();
    let expected = a.mem_info().unwrap().1;
    let b = CudaBackend::new(1).unwrap();
    let observed = a.mem_info().unwrap().1;
    let mem = a.alloc(0, 16).unwrap();
    // The test driver records the actual context immediately before the allocation.
    let actual_context = unsafe { *((mem.base as *const u64).sub(1)) };
    drop(mem);
    drop(b);
    assert_eq!(
        (observed, actual_context),
        (expected, 1),
        "backend A read GPU B's memory and allocated in B's context"
    );
}

#[test]
fn dropping_other_gpu_allocation_does_not_invalidate_cached_context() {
    let Some(_test) = setup() else { return };
    let a = CudaBackend::new(0).unwrap();
    let b = CudaBackend::new(1).unwrap();
    let b_mem = b.alloc(0, 16).unwrap();
    let expected = a.mem_info().unwrap().1;
    drop(b_mem);
    assert_eq!(
        a.mem_info().unwrap().1,
        expected,
        "CudaFreer changed the context without updating LAST_CTX"
    );
}

/// Build the mock driver once and take the suite lock, or `None` when this
/// host has no C compiler.
///
/// Skipping rather than failing matches every other environment-dependent test
/// here (the `PLOW_GPU_TEST` suites print a reason and return). Asserting on
/// `cc` turned "this image has no compiler" into a red suite instead of a
/// skipped one.
fn setup() -> Option<std::sync::MutexGuard<'static, ()>> {
    static TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let test = TEST.lock().unwrap();
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*READY.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("plow-cuda-context-{}", std::process::id()));
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        let lib = dir.join("libmock_cuda.so");
        let built = std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/common/mock_cuda.c"
            ))
            .arg("-o")
            .arg(&lib)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !built {
            return false;
        }
        let mut cfg = plowrt::config::RuntimeConfig::get().clone();
        cfg.nv.libcuda = Some(lib.to_string_lossy().into_owned());
        plowrt::config::RuntimeConfig::init(cfg);
        true
    }) {
        eprintln!("skipped: no C compiler to build the mock CUDA driver");
        return None;
    }
    Some(test)
}

#[test]
fn reopening_a_context_released_on_another_thread_rebinds_a_recycled_handle() {
    let Some(_test) = setup() else { return };
    let a = CudaBackend::new(0).unwrap();
    let expected = a.mem_info().unwrap().1;
    std::thread::spawn(move || drop(a)).join().unwrap();
    let b = CudaBackend::new(0).unwrap();
    assert_eq!(b.mem_info().unwrap().1, expected);
}
