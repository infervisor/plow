//! Static analysis of compiled programs — no device, no execution.
//!
//! [`graph`] is the dependency-graph census that `examples/graphstat.rs` used to
//! own; [`counters`] is the per-counter detail layered above it.

pub mod counters;
pub mod graph;
