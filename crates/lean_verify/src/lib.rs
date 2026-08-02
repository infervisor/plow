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

impl VerifyError {
    /// True when the failure means "there is no usable `plow_verify` on this
    /// machine", as opposed to "the verifier looked at this program and said
    /// no".
    ///
    /// THIS PREDICATE IS THE SAFETY MECHANISM for default-on verification, not
    /// [`binary_available`]. A caller that runs the gate by default must
    /// downgrade exactly this class to a warning and record the skip; every
    /// other error, and every `ok == false` certificate, is a real finding and
    /// must fail loudly.
    ///
    /// The three shapes it covers, all observed in this repo:
    ///   * `BinaryNotFound` — `PLOW_VERIFY_BIN` points at a non-file.
    ///   * `Spawn` — nothing at that path / no `+x` bit / wrong ELF class.
    ///     `locate_binary` falls back to the bare name `plow_verify`, so a
    ///     plain "not on PATH" arrives here and NOT as `BinaryNotFound`.
    ///   * a deserialize failure with EMPTY stdout — the process started and
    ///     produced nothing. `lean-plow`'s binary links a `/nix/store` ELF
    ///     interpreter, so outside `nix develop` it dies with exit 127 before
    ///     `main`; that is an unusable binary, not a protocol error. A genuine
    ///     rejection always comes back as JSON, so a NON-empty stdout that
    ///     fails to parse stays a hard error.
    pub fn is_binary_unusable(&self) -> bool {
        match self {
            VerifyError::BinaryNotFound(_) | VerifyError::Spawn(_) => true,
            VerifyError::DeserializeCertificate(_, out)
            | VerifyError::DeserializeQueryResult(_, out) => out.trim().is_empty(),
            _ => false,
        }
    }
}

/// Cheap probe: does something that looks like `plow_verify` exist here?
///
/// **THIS IS AN OPTIMISATION, NOT THE SAFETY MECHANISM.** `is_file()` (or a
/// PATH hit) does not mean runnable: wrong architecture, no `+x` bit, missing
/// Lean runtime shared objects, or the `/nix/store` ELF-interpreter trap above
/// all pass this probe and then fail at spawn. The real net is the caller
/// downgrading [`VerifyError::is_binary_unusable`] to a warning.
///
/// What this probe is FOR: letting a default-on caller skip the *preparation*
/// work whose only consumer is the verifier — the egglog fusion report, and
/// marshaling an entire GQ stream into a `ScheduleRequest` — instead of
/// building it and throwing it away.
///
/// DO NOT "simplify" this by deleting the error downgrade and trusting the
/// probe. A present-but-broken binary would then take every compile from
/// "warn and skip" to "hard failure", which is precisely the regression this
/// split exists to prevent.
pub fn binary_available() -> bool {
    let Ok(p) = locate_binary() else { return false };
    if p.components().count() > 1 {
        return p.is_file();
    }
    // Bare name: `locate_binary`'s PATH fallback. Resolve it the way `Command`
    // would, so the probe answers the same question the spawn will ask.
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|d| d.join(&p).is_file())
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
    let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../lean-plow/.lake/build/bin/plow_verify");
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
    // Checkpoints D and F run the IDENTICAL Lean computation on the identical
    // `ScheduleRequest` bundle (memory.rs: "callers hand the exact same
    // bundle"); only the certificate's message text differs. Cache the
    // verdict by payload hash so the second spawn (minutes of reachability
    // work on a full-model bucket) is free. Scoped to the D/F handler pair —
    // any other checkpoint always spawns.
    let df_cache_key = if checkpoint == "D" || checkpoint == "F" {
        use std::hash::{Hash, Hasher};
        let bytes = serde_json::to_vec(&payload).map_err(VerifyError::SerializeRequest)?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        Some(h.finish())
    } else {
        None
    };
    static DF_CACHE: std::sync::Mutex<Option<(u64, bool, Option<String>)>> =
        std::sync::Mutex::new(None);
    if let Some(key) = df_cache_key {
        if let Some((k, ok, reason)) = DF_CACHE.lock().unwrap().as_ref() {
            if *k == key {
                return Ok(Certificate {
                    ok: *ok,
                    checkpoint: checkpoint.to_string(),
                    notes: Some("verdict cached from the identical D/F payload".into()),
                    reason: reason.clone(),
                });
            }
        }
    }
    let request = serde_json::json!({
        "checkpoint": checkpoint,
        "payload": payload,
    });
    // Debug aid: `PLOW_VERIFY_DUMP=<dir>` writes every request to a file so a
    // subprocess failure ("EOF while parsing" = the Lean side died with empty
    // stdout) can be replayed against `plow_verify` by hand.
    if let Ok(dir) = std::env::var("PLOW_VERIFY_DUMP") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = format!("{dir}/plow-verify-{n:03}-{checkpoint}.json");
        if let Ok(bytes) = serde_json::to_vec(&request) {
            let _ = std::fs::write(path, bytes);
        }
    }
    let stdout = invoke(&request)?;
    let cert: Certificate = serde_json::from_str(stdout.trim())
        .map_err(|e| VerifyError::DeserializeCertificate(e, stdout.clone()))?;
    if let Some(key) = df_cache_key {
        *DF_CACHE.lock().unwrap() = Some((key, cert.ok, cert.reason.clone()));
    }
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
            result
                .error
                .unwrap_or_else(|| "query returned ok=false with no error".into()),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of [`VerifyError::is_binary_unusable`]: it must split
    /// "there is no working verifier here" (degrade, warn, record the skip)
    /// from "the verifier read this program and said no" (fail loudly).
    ///
    /// Wrong in the permissive direction and a REJECTION — a real bug caught —
    /// becomes a warning nobody reads.
    #[test]
    fn only_unusable_binary_failures_are_downgradable() {
        let io = || std::io::Error::from(std::io::ErrorKind::NotFound);
        assert!(VerifyError::BinaryNotFound("/nope".into()).is_binary_unusable());
        assert!(VerifyError::Spawn(io()).is_binary_unusable());

        // A rejection is a FINDING, never a skip.
        assert!(!VerifyError::Rejected("cycle in ordering graph".into()).is_binary_unusable());
        assert!(!VerifyError::QueryFailed("bad request".into()).is_binary_unusable());
        assert!(!VerifyError::NotImplemented("C").is_binary_unusable());
    }

    /// The exit-127 trap: `lean-plow`'s binary links a `/nix/store` ELF
    /// interpreter, so outside `nix develop` it dies before `main` and returns
    /// EMPTY stdout. That is an unusable binary. Stdout with content that does
    /// not parse is a protocol error and stays hard — a rejection always
    /// arrives as JSON.
    #[test]
    fn empty_stdout_is_an_unusable_binary_but_garbage_stdout_is_not() {
        let bad = || serde_json::from_str::<Certificate>("x").unwrap_err();
        assert!(VerifyError::DeserializeCertificate(bad(), String::new()).is_binary_unusable());
        assert!(VerifyError::DeserializeCertificate(bad(), "  \n".into()).is_binary_unusable());
        assert!(
            !VerifyError::DeserializeCertificate(bad(), "{\"ok\":".into()).is_binary_unusable()
        );
    }

    /// The probe is an OPTIMISATION and may be wrong optimistically (a
    /// present-but-unrunnable binary passes it — that is what the downgrade
    /// above is for). It must not be wrong the other way for the one case it
    /// can answer exactly: an explicit `PLOW_VERIFY_BIN` that is not a file.
    #[test]
    fn an_explicit_binary_path_that_does_not_exist_fails_the_probe() {
        // The only test in this crate touching this var; `locate_binary` reads
        // it directly, so there is nothing to inject it through.
        std::env::set_var("PLOW_VERIFY_BIN", "/nonexistent/plow_verify");
        assert!(!binary_available());
        assert!(matches!(
            locate_binary(),
            Err(VerifyError::BinaryNotFound(_))
        ));
        std::env::remove_var("PLOW_VERIFY_BIN");
    }
}
