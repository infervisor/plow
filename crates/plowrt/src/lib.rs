//! # plowrt — the plow host runtime
//!
//! `plowc` is a compiler: it lowers a model for a GPU into per-bucket **assets**
//! (`.pkt` packet streams, `.map.json` address maps, `weights.json`, sidecars).
//! `plowrt` is the host that *runs* those assets. It:
//!
//! * loads a compiled [`asset::ModelBundle`] from disk ([`asset`]),
//! * drives a device through the [`device::Backend`] trait — a real CPU
//!   reference backend ships here; CUDA/HSA are FFI backends behind features,
//! * lays weights and the KV cache into HBM ([`memory`]) and streams them
//!   in/out under pressure ([`memory::streamer`]),
//! * launches the persistent-kernel executors once and coordinates them through
//!   counter pools, packet queues, and a bidirectional OOB channel ([`exec`]),
//! * schedules per-iteration work with queuing-theory admission/batching
//!   ([`sched`]),
//! * routes requests by model slug through single- or multi-model pipelines
//!   ([`orch`]),
//! * and serves an OpenAI-compatible API ([`serve`]).
//!
//! The design is documented per subsystem under `docs/runtime/` and in
//! `plans/design-the-runtime-in-whimsical-leaf.md`.

pub mod asset;
pub mod device;
pub mod exec;
pub mod memory;
pub mod obs;
pub mod orch;
pub mod sched;
pub mod serve;
pub mod sim;
pub mod text;

mod error;
pub use error::{Result, RuntimeError};

/// Declare a cached environment-variable boolean flag. The first call reads
/// `std::env::var(VAR)` and caches the result in a `OnceLock`; subsequent calls
/// are a single atomic load. Use for hot-path feature gates.
///
/// ```ignore
/// env_flag!(fn my_flag, "PLOW_MY_FLAG");
/// if my_flag() { … }
/// ```
macro_rules! env_flag {
    // Explicit visibility: env_flag!(pub fn x, ..) / env_flag!(pub(crate) fn x, ..).
    (pub $(($vis:tt))? fn $name:ident, $var:expr) => {
        pub $(($vis))? fn $name() -> bool {
            static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *FLAG.get_or_init(|| std::env::var($var).map(|v| v == "1").unwrap_or(false))
        }
    };
    // Default (bare `fn`): `pub(crate)`, not private — `serve::mux` invokes this INSIDE a
    // `mod packlog { … }` and calls `packlog::on()` from the parent module, which a private fn
    // rejects (E0603). Only the cuda-gated call sites hit it, so the default build never caught it.
    // Still crate-internal, matching the `pub(crate) use env_flag` below.
    (fn $name:ident, $var:expr) => {
        pub(crate) fn $name() -> bool {
            static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *FLAG.get_or_init(|| std::env::var($var).map(|v| v == "1").unwrap_or(false))
        }
    };
}
pub(crate) use env_flag;
