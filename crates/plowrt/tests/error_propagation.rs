//! Error-propagation surface tests that need no GPU: the typed device-fault
//! API a client of plowrt sees. The pieces that DO need hardware — a real
//! kernel trap poisoning the CUDA context / HSA queue, bind()/guard()
//! short-circuiting live driver calls, and the mux rejecting with 503 after a
//! real fault — are covered by unit tests of the pure logic in-module
//! (`device::cuda`/`device::hsa` fatal tables, `serve::mux` EngineHealth,
//! `serve` status_for) and remain to be exercised end-to-end under
//! PLOW_GPU_TEST on a box with a device.

use plowrt::{DeviceErrorInfo, RuntimeError};

fn fault(code: i32, name: &str, fatal: bool) -> RuntimeError {
    RuntimeError::DeviceFault {
        info: DeviceErrorInfo {
            operation: "cuStreamSynchronize".into(),
            code,
            name: name.into(),
            fatal,
        },
    }
}

#[test]
fn device_fault_carries_the_numeric_code_through_display() {
    let e = fault(700, "CUDA_ERROR_ILLEGAL_ADDRESS", true);
    let s = e.to_string();
    assert!(s.contains("cuStreamSynchronize"), "{s}");
    assert!(s.contains("CUDA_ERROR_ILLEGAL_ADDRESS"), "{s}");
    assert!(s.contains("code 700"), "{s}");
}

#[test]
fn fatality_is_queryable_without_string_matching() {
    assert!(fault(719, "CUDA_ERROR_LAUNCH_FAILED", true).is_fatal());
    assert!(!fault(2, "CUDA_ERROR_OUT_OF_MEMORY", false).is_fatal());
    assert_eq!(
        fault(719, "CUDA_ERROR_LAUNCH_FAILED", true).device_code(),
        Some(719)
    );
    assert_eq!(
        fault(719, "CUDA_ERROR_LAUNCH_FAILED", true).device_operation(),
        Some("cuStreamSynchronize")
    );
}

#[test]
fn legacy_variants_expose_no_device_fault() {
    for e in [
        RuntimeError::Device("validation message".into()),
        RuntimeError::Oom("kv".into()),
        RuntimeError::Rejected("shed".into()),
        RuntimeError::Msg("x".into()),
    ] {
        assert!(e.device_fault().is_none(), "{e}");
        assert!(!e.is_fatal(), "{e}");
        assert_eq!(e.device_code(), None, "{e}");
    }
}
