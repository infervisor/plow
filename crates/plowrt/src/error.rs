//! Runtime error type. Errors live off the hot path — request setup, asset
//! loading, device bringup — so `thiserror` + boxing is fine here; the per-token
//! path returns plain values and never allocates an error.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// A failed driver call, with the numeric status the driver returned.
///
/// `fatal` marks statuses that permanently poison the device context (a
/// trapped kernel, a destroyed context): once one is seen the backend stops
/// dispatching and every later call short-circuits with a clone of this info.
/// Built on the error path only — the strings allocate at construction time,
/// never in the success case.
#[derive(Debug, Clone)]
pub struct DeviceErrorInfo {
    /// The driver call that failed, e.g. `"cuStreamSynchronize"`.
    pub operation: String,
    /// The raw status: a `CUresult` or `hsa_status_t`.
    pub code: i32,
    /// The driver's name for the code, e.g. `"CUDA_ERROR_LAUNCH_FAILED"`.
    pub name: String,
    /// Does this status permanently poison the device context?
    pub fatal: bool,
}

impl std::fmt::Display for DeviceErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} (code {})", self.operation, self.name, self.code)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("asset io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("json parse error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("malformed packet stream {path}: {reason}")]
    Packet { path: PathBuf, reason: String },

    #[error("invalid address map: {0}")]
    AddressMap(String),

    #[error("device error: {0}")]
    Device(String),

    /// A driver call failed with a status code. [`Device`](Self::Device) stays
    /// for validation/non-driver errors; this variant carries the numeric code
    /// so logs and HTTP mapping can act on it.
    #[error("device fault: {info}")]
    DeviceFault { info: DeviceErrorInfo },

    #[error("no model registered for slug '{0}'")]
    UnknownModel(String),

    #[error("out of device memory: {0}")]
    Oom(String),

    #[error("deadlock detected: {0}")]
    Deadlock(String),

    #[error("request rejected: {0}")]
    Rejected(String),

    #[error("{0}")]
    Msg(String),
}

impl RuntimeError {
    /// The driver fault behind this error, when there is one.
    pub fn device_fault(&self) -> Option<&DeviceErrorInfo> {
        match self {
            RuntimeError::DeviceFault { info } => Some(info),
            _ => None,
        }
    }

    /// The raw driver status code (`CUresult` / `hsa_status_t`), when known.
    pub fn device_code(&self) -> Option<i32> {
        self.device_fault().map(|i| i.code)
    }

    /// The driver call that failed, when known.
    pub fn device_operation(&self) -> Option<&str> {
        self.device_fault().map(|i| i.operation.as_str())
    }

    /// Does this error mean the device context is permanently poisoned?
    pub fn is_fatal(&self) -> bool {
        self.device_fault().is_some_and(|i| i.fatal)
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceErrorInfo, RuntimeError};

    fn fault(fatal: bool) -> RuntimeError {
        RuntimeError::DeviceFault {
            info: DeviceErrorInfo {
                operation: "cuStreamSynchronize".into(),
                code: 700,
                name: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
                fatal,
            },
        }
    }

    #[test]
    fn device_fault_display_names_op_code_and_name() {
        assert_eq!(
            fault(true).to_string(),
            "device fault: cuStreamSynchronize: CUDA_ERROR_ILLEGAL_ADDRESS (code 700)"
        );
    }

    #[test]
    fn helpers_expose_code_op_and_fatality() {
        let e = fault(true);
        assert_eq!(e.device_code(), Some(700));
        assert_eq!(e.device_operation(), Some("cuStreamSynchronize"));
        assert!(e.is_fatal());
        assert!(!fault(false).is_fatal());
        // Non-driver errors expose nothing and are never fatal.
        let plain = RuntimeError::Device("validation".into());
        assert_eq!(plain.device_code(), None);
        assert!(!plain.is_fatal());
    }
}
