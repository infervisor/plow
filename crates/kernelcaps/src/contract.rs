//! Block contracts for kernel qualification: costmodel's [`BlockContract`], so the registry,
//! tunedb's certification gate and the campaign floors price an op with one model and one
//! hardware table (`hwspec::peak_compute_tflops`).
pub use costmodel::contract::{BlockContract, BoundCategory};
