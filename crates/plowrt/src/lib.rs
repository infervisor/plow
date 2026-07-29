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

pub mod analysis;
pub mod asset;
pub mod device;
pub mod disasm;
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
    // Default (bare `fn`): `pub(crate)`, not private. The case that forced this was
    // `serve::mux`'s `mod packlog`, which invoked the macro inside a child module and called
    // `packlog::on()` from the parent — a private fn rejects that (E0603), and only the
    // cuda-gated call sites hit it, so the default build never caught it. That module is gone,
    // but the visibility stays: the same shape recurs the moment any flag is declared inside a
    // submodule, and `pub(crate)` is the correct scope for a crate-internal flag regardless.
    (fn $name:ident, $var:expr) => {
        pub(crate) fn $name() -> bool {
            static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *FLAG.get_or_init(|| std::env::var($var).map(|v| v == "1").unwrap_or(false))
        }
    };
}
pub(crate) use env_flag;

/// Declare a cached environment-variable `usize` knob — the numeric sibling of
/// [`env_flag!`], with the same one-`OnceLock`-per-knob, read-once discipline.
///
/// The `zero = unbounded` form encodes the convention every rows/budget knob in
/// the serve path already used independently: `PLOW_PF_INTERLEAVE=0` means "no
/// bound", not "no rows", so `0` caches as `usize::MAX` and the call site stays
/// a plain comparison on the hot path.
///
/// ```ignore
/// env_usize!(fn chunk_cost_rows, "PLOW_PF_CHUNK_COST", default 512);
/// env_usize!(fn interleave_rows, "PLOW_PF_INTERLEAVE", default 2048, zero = unbounded);
/// ```
// Every knob declared with this today lives on the CUDA prefill path, so a
// build without `--features cuda` legitimately has no user. Not cfg-gated on
// the feature: the macro itself is backend-agnostic and the AMD path's own
// numeric knobs belong here as they are added.
#[allow(unused_macros)]
macro_rules! env_usize {
    (fn $name:ident, $var:expr, default $d:expr, zero = unbounded) => {
        pub(crate) fn $name() -> usize {
            static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *V.get_or_init(|| {
                match std::env::var($var).ok().and_then(|v| v.parse::<usize>().ok()) {
                    Some(0) => usize::MAX,
                    Some(n) => n,
                    None => $d,
                }
            })
        }
    };
    (fn $name:ident, $var:expr, default $d:expr) => {
        pub(crate) fn $name() -> usize {
            static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *V.get_or_init(|| {
                std::env::var($var).ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or($d)
            })
        }
    };
}
#[allow(unused_imports)]
pub(crate) use env_usize;
