//! §F Pipeline — an ordered chain of model stages with zero-copy dataflow.
//!
//! Single-model text is a one-stage pipeline; multimodal is
//! `image → vision_enc → projector → text_decode`, where a stage's output buffer
//! aliases the next stage's input slot (via the address map). Different requests'
//! stages pipeline across executors — the counter graph already encodes ordering.
//!
//! ## Design (plow runtime)
//!
//! Inter-stage data movement is compiled into device packets — the host never
//! touches intermediate activations. The counter graph encodes the ordering:
//! stage N's final packet fires a counter that gates stage N+1's first packet.
//! This means `run_tick` iterates stages only for bookkeeping (slot routing,
//! scheduling metadata); the actual data flow is device-side.

use std::sync::Arc;

use crate::asset::{ModelBundle, Phase};

/// One stage: a bundle bound to a phase (and, on real HW, to an executor pool).
#[derive(Clone)]
pub struct Stage {
    pub bundle: Arc<ModelBundle>,
    pub phase: Phase,
    /// Human label for tracing (`"vision_enc"`, `"decode"`, `"draft"`, …).
    pub label: String,
    /// Device id this stage is pinned to (TP/PP placement). `None` = same
    /// device as the previous stage (collocated).
    pub device: Option<u8>,
}

/// An ordered pipeline plan.
#[derive(Clone, Default)]
pub struct Pipeline {
    pub stages: Vec<Stage>,
}

impl Pipeline {
    /// A one-stage text pipeline.
    pub fn single(bundle: Arc<ModelBundle>, phase: Phase) -> Self {
        Pipeline {
            stages: vec![Stage {
                bundle,
                phase,
                label: "decode".into(),
                device: None,
            }],
        }
    }

    pub fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    pub fn is_multimodal(&self) -> bool {
        self.stages.len() > 1
    }

    /// Number of stages in the pipeline.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Run one tick across all pipeline stages. In the plow model, inter-stage
    /// transfers are compiled device packets (counter-gated), so this method
    /// handles only the scheduling bookkeeping: iterating stages, selecting
    /// the right bucket per stage, and propagating slot metadata.
    ///
    /// Returns the primary (last-stage) bundle for the mux's token-production
    /// path. Single-stage pipelines short-circuit to `&stages[0].bundle`.
    pub fn primary_bundle(&self) -> Option<&Arc<ModelBundle>> {
        self.stages.last().map(|s| &s.bundle)
    }

    /// For multi-stage: returns an iterator over (stage_index, bundle, phase)
    /// that the mux can use to drive prefill across encoder → decoder.
    pub fn stage_iter(&self) -> impl Iterator<Item = (usize, &Arc<ModelBundle>, Phase)> {
        self.stages
            .iter()
            .enumerate()
            .map(|(i, s)| (i, &s.bundle, s.phase))
    }
}
