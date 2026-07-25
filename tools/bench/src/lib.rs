//! Version-pin anchor for the `inference-benchmarker` git dependency.
//!
//! Intentionally empty: this crate exists so the workspace `Cargo.lock` pins
//! the benchmark harness rev (see `Cargo.toml` in this directory). The
//! benchmark itself runs through the tool's own binary, built from the same
//! rev by `perf-data/bench_ib.sh`.
