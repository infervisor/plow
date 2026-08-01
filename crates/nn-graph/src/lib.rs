//! `nn-graph` — symbolic neural-network operator graph IR.
//!
//! Stage-1 frontend IR from the Infervisor compiler design. This crate is
//! model-agnostic: it builds full operator graphs, runs symbolic shape
//! inference, specializes shapes to concrete batch/resolution parameters,
//! visualizes the network, and enumerates the per-op weights a loader parses at
//! later stages. Architecture-specific builders (which HF model → which ops)
//! live in [`models`], behind the `models` feature. Resolving a model *id* to
//! its `config.json` over the network is [`hub`], behind the `hub` feature.
//!
//! Pipeline position:
//!
//! ```text
//! GraphBuilder/Nn ──▶ Graph ──(infer_shapes)──▶ symbolic Graph
//!                                   │
//!                          (bind: B/S/L params)──▶ concrete Graph
//!                                   │
//!                            weight_manifest
//! ```
//!
//! # Where this sits in plow, honestly
//!
//! Two consumers, very unequal:
//!
//! * **[`DType`] is load-bearing.** `costmodel::dtype_cost` maps it to MMA rates
//!   and SRAM staging, and every `rewrite::OpSpec` carries a weight/compute
//!   dtype pair. This is the part of the crate that reaches emitted code.
//! * **[`Graph`] has no shipping consumer.** `rewrite` lowers it to egglog and
//!   `rewrite::plan_from_all_blocks` builds a `LayerPlan` from it, but the
//!   fused graph is computed for statistics and dropped — see the `plowc` crate
//!   docs and `perf-data/px18-egglog-wholemodel.md` (measured on Gemma-4-12B: 0
//!   of 1156 ops, 0 of 24,226 GFLOP). The `--emit devblob` path that produces
//!   GPU-executable output goes `plowc --hf-dir` → `hf_config` → `devgen`, and
//!   `devgen` does not depend on this crate at all.
//!
//! So: a change to [`DType`] can change emitted packets. A change to a builder
//! in [`models`] changes analysis output and nothing else. Know which one you
//! are editing.

pub mod bind;
pub mod builder;
pub mod dim;
pub mod dtype;
pub mod graph;
#[cfg(feature = "hub")]
pub mod hub;
pub mod infer;
#[cfg(feature = "models")]
pub mod models;
pub mod op;
pub mod shape;
#[cfg(feature = "models")]
pub mod viz;

pub use bind::Bindings;
pub use builder::Nn;
pub use dim::{Dim, SymId, SymbolTable};
pub use dtype::DType;
pub use graph::{Graph, GraphBuilder, Node, NodeId, Origin, TensorId, TensorInfo, WeightSpec};
pub use infer::{infer_shapes, InferError};
pub use op::{ActKind, EwKind, LinearAttnKind, MoeGroups, Op, ReduceKind};
pub use shape::Shape;
