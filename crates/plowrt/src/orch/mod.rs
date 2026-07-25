//! §F Multi-model registry + multimodal pipeline, plus the §L multi-model
//! features (MoE dispatch, prefill/decode disaggregation, speculative decoding
//! as a two-model pipeline).

pub mod disagg;
pub mod moe;
pub mod pipeline;
pub mod registry;
pub mod router;
pub mod speculative;
pub mod transport;

pub use pipeline::{Pipeline, Stage};
pub use registry::Registry;
pub use router::route;
pub use transport::{DeviceAddr, NullTransport, Transport, TransferDescriptor};
