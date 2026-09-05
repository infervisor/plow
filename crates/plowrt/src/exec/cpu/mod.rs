//! CPU execution engine — the device-ISA interpreter on persistent, core-pinned
//! worker threads, with kernels in C (`runtime/cpu/dev/`) reached through [`ffi`].
//!
//! Plan: `plans/cpu-backend.md`. Module ownership during bring-up:
//! * [`ffi`] — bindings + ABI lock (P0). Gated on the `cpu` feature because it
//!   links the C library built by `build.rs`.
//! * [`topology`], [`control`], [`workers`], [`interp`] — pure Rust (P1), so they
//!   build and test without the C library.

#[cfg(feature = "cpu")]
pub mod ffi;

pub mod control;
pub mod interp;
pub mod topology;
pub mod workers;
