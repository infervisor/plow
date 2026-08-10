//! Version-pin anchor for the `inference-benchmarker` git dependency.
//!
//! Intentionally empty: this crate exists so the workspace `Cargo.lock` pins
//! the benchmark harness rev (see `Cargo.toml` in this directory). The
//! benchmark itself runs through the tool's own binary, installed from the
//! same rev (`cargo install --git … --rev …`); see
//! `docs/bringup/07-perf-campaign.md`.
