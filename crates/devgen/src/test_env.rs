//! Env isolation for the src-side unit tests.
//!
//! Several emit knobs are read LIVE from the environment on every call — e.g.
//! `mla::wgfit` reads `PLOW_GLM_WGFIT` per invocation, by design, so an A/B
//! control arm comes out of the same `plowc` binary as the fixed one. Env is
//! process-global and cargo runs a crate's unit tests as threads of ONE
//! process, so a test that sets such a knob changes what every concurrently
//! running test emits.
//!
//! That is not hypothetical: `the_prefill_ladder_leaves_the_decode_program_byte_identical`
//! failed roughly one full-suite run in seven with `decode inst 77: block count
//! left: 256, right: 32` — the un-narrowed width it sees during the ~3-line
//! window when a sibling test holds `PLOW_GLM_WGFIT=0`. A flaky regression net
//! is worse than none, because the next real regression reads as noise.
//!
//! Same shape as `tests/golden_blob.rs`'s `EMIT_LOCK`/`EnvScope`, which cannot
//! be shared with this side (integration tests are a separate crate).
//!
//! Discipline: any test that SETS one of these knobs, and any test that emits a
//! program whose shape depends on one, takes [`env_guard`] for its whole body.

/// Serializes every test that touches a live-read emit knob.
///
/// Poisoning is irrelevant here — a panicking test has already failed — so the
/// guard is unwrapped through rather than propagating a second failure.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets env vars for a scope and restores the previous values on drop —
/// including the `None` case, so a test cannot leak a knob into whatever runs
/// next. Restoring on unwind is the point: a `#[should_panic]` test that sets a
/// pin would otherwise leave it set for the rest of the process.
pub(crate) struct EnvScope(Vec<(String, Option<String>)>);

impl EnvScope {
    pub(crate) fn set(kv: &[(&str, &str)]) -> Self {
        let saved = kv
            .iter()
            .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
            .collect();
        for (k, v) in kv {
            std::env::set_var(k, v);
        }
        resnapshot();
        EnvScope(saved)
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        resnapshot();
    }
}

/// Re-read the environment into the active [`crate::emit_config::EmitConfig`].
///
/// REQUIRED after any mutation, and easy to forget. The knobs this module was written for
/// used to be read live on every call, so setting the var WAS the whole operation. Most are
/// now `EmitConfig` fields, and `emit_config::active()` caches — the first reader in the
/// process pins the snapshot, so without this a scoped `set` would change the environment
/// and nothing else.
///
/// That caching is correct for production (one `plowc --emit` process = one config, and an
/// A/B arm is a separate process), and wrong only for the in-process case this module exists
/// to serve. `install` is built for it: last call wins, and the leaked box is bounded by the
/// number of scopes.
fn resnapshot() {
    crate::emit_config::install(crate::emit_config::EmitConfig::from_env());
}
