//! Runtime error type. Errors live off the hot path — request setup, asset
//! loading, device bringup — so `thiserror` + boxing is fine here; the per-token
//! path returns plain values and never allocates an error.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, RuntimeError>;

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
