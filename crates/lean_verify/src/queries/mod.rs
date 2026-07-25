//! Typed query interfaces for the Lean performance oracle.
//!
//! Each sub-module defines the request/response types for one query kind.
//! The caller builds a request struct, calls the corresponding `check_*`
//! function, and gets back a typed result (not raw JSON).

pub mod counter_granularity;
pub mod lower_bound;
