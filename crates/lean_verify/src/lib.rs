//! Rust client for the `plow_verify` Lean 4 CLI.
//!
//! The Lean side (`lean-plow/`) proves the universal lemmas that back plow's
//! per-instance safety checks. Callers here spawn `plow_verify`, send a JSON
//! request, and get a [`Certificate`] back.
//!
//! Six checkpoints correspond to the pipeline stages in
//! `plans/lean-formal-verification-analysis.md §5.10`:
//!
//! | ID | Stage         | Status     |
//! |----|---------------|------------|
//! | A  | Rewrite       | stub       |
//! | B  | Assemble      | stub       |
//! | C  | Collapse/Relax| stub       |
//! | D  | Schedule      | wired      |
//! | E  | Emit          | stub       |
//! | F  | Memory        | wired (shares D's verifier) |
//!
//! Additionally, the `query` interface allows the compiler to ask Lean for
//! provably-optimal decisions (counter granularity, lower bounds, ordering
//! quality) rather than reimplementing the logic in Rust. See
//! `plans/lean-active-perf-impl.md`.
//!
//! The `PLOW_VERIFY_BIN` env var overrides the binary path (default:
//! `plow_verify` looked up on `PATH`, then `lean-plow/.lake/build/bin/plow_verify`
//! relative to the crate root).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

pub mod checkpoints;
pub mod queries;

/// Certificate returned by the Lean verifier for a single checkpoint call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub ok: bool,
    pub checkpoint: String,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Result of a performance query to Lean. Unlike [`Certificate`] (accept/reject),
/// a query returns a computed answer with an optional correctness certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Whether the query succeeded (could be parsed + computed).
    pub ok: bool,
    /// The query type that was dispatched.
    pub query: String,
    /// The computed answer — structure depends on query type.
    pub answer: serde_json::Value,
    /// Human-readable certificate referencing the backing theorem.
    #[serde(default)]
    pub certificate: Option<String>,
    /// Error reason if `ok` is false.
    #[serde(default)]
    pub error: Option<String>,
    /// Wall-clock time in milliseconds the Lean side took.
    #[serde(default)]
    pub time_ms: Option<u64>,
}

/// Failure modes when talking to `plow_verify`.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("plow_verify binary not found (set PLOW_VERIFY_BIN or build lean-plow): {0}")]
    BinaryNotFound(String),
    #[error("spawn failed: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("could not serialize request: {0}")]
    SerializeRequest(serde_json::Error),
    #[error("could not deserialize certificate: {0} (stdout was: {1})")]
    DeserializeCertificate(serde_json::Error, String),
    #[error("could not deserialize query result: {0} (stdout was: {1})")]
    DeserializeQueryResult(serde_json::Error, String),
    #[error("verifier rejected: {0}")]
    Rejected(String),
    #[error("query failed: {0}")]
    QueryFailed(String),
    #[error("checkpoint {0} not yet implemented on the Lean side")]
    NotImplemented(&'static str),
}

fn locate_binary() -> Result<PathBuf, VerifyError> {
    if let Ok(p) = std::env::var("PLOW_VERIFY_BIN") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(VerifyError::BinaryNotFound(path.display().to_string()));
    }
    // Try relative to crate — most useful in tests.
    let candidate =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../lean-plow/.lake/build/bin/plow_verify");
    if candidate.is_file() {
        return Ok(candidate);
    }
    // Fall back to PATH lookup by leaving it as a bare name; Command handles that.
    Ok(PathBuf::from("plow_verify"))
}

/// Low-level: send a JSON request to `plow_verify` and return raw stdout.
fn invoke(request: &serde_json::Value) -> Result<String, VerifyError> {
    let request_bytes = serde_json::to_vec(request).map_err(VerifyError::SerializeRequest)?;

    let bin = locate_binary()?;
    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // Write stdin from a thread while draining stdout/stderr on this one:
    // checkpoint-D payloads scale with task count, and a blocking `write_all`
    // against a child that emits output before consuming its input deadlocks
    // both sides on full pipes.
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let writer = std::thread::spawn(move || stdin.write_all(&request_bytes));
    let output = child.wait_with_output()?;
    match writer.join() {
        Ok(Ok(())) => {}
        // Child exited before reading all input — its output (parsed below)
        // carries the real diagnostic.
        Ok(Err(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
        Ok(Err(e)) => return Err(VerifyError::Spawn(e)),
        Err(_) => {
            return Err(VerifyError::Spawn(std::io::Error::other(
                "stdin writer thread panicked",
            )))
        }
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Send a verification request to `plow_verify` and parse the certificate.
pub fn call(checkpoint: &str, payload: serde_json::Value) -> Result<Certificate, VerifyError> {
    let request = serde_json::json!({
        "checkpoint": checkpoint,
        "payload": payload,
    });
    let stdout = invoke(&request)?;
    let cert: Certificate = serde_json::from_str(stdout.trim())
        .map_err(|e| VerifyError::DeserializeCertificate(e, stdout.clone()))?;
    Ok(cert)
}

/// Send a performance query to `plow_verify` and parse the result.
///
/// Queries differ from checkpoints: they compute an answer (not just accept/reject)
/// and return it with a correctness certificate. The Lean side dispatches based on
/// the `query` field.
///
/// # Example
///
/// ```ignore
/// let result = lean_verify::query("counter_granularity", serde_json::json!({
///     "edges": [{"id": 7, "consumer_slices": [0,1,2,3], "work": [100,100,100,100]}]
/// }))?;
/// // result.answer contains per-edge fine/coarse decisions
/// ```
pub fn query(query_type: &str, payload: serde_json::Value) -> Result<QueryResult, VerifyError> {
    let request = serde_json::json!({
        "query": query_type,
        "payload": payload,
    });
    let stdout = invoke(&request)?;
    let result: QueryResult = serde_json::from_str(stdout.trim())
        .map_err(|e| VerifyError::DeserializeQueryResult(e, stdout.clone()))?;
    if !result.ok {
        return Err(VerifyError::QueryFailed(
            result.error.unwrap_or_else(|| "query returned ok=false with no error".into()),
        ));
    }
    Ok(result)
}

/// Convenience: call and turn a `!ok` cert into an error.
pub fn require(checkpoint: &str, payload: serde_json::Value) -> Result<Certificate, VerifyError> {
    let cert = call(checkpoint, payload)?;
    if cert.ok {
        Ok(cert)
    } else {
        Err(VerifyError::Rejected(
            cert.reason.unwrap_or_else(|| "no reason".into()),
        ))
    }
}
