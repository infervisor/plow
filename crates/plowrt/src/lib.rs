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
//! The design is documented per subsystem under `docs/runtime/`.

pub mod analysis;
pub mod asset;
pub mod config;
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
pub use error::{DeviceErrorInfo, Result, RuntimeError};
