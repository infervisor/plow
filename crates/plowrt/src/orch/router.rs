//! §F Router — request `model` slug / workflow id → pipeline plan.

use std::sync::Arc;

use crate::asset::{ModelBundle, Phase};
use crate::orch::pipeline::{Pipeline, Stage};

/// Whether the request carries non-text content parts (drives multimodal
/// prefixing with a vision stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    /// Has at least one image content part.
    VisionText,
}

/// Build the pipeline for a resolved model. Text → single decode stage;
/// vision+text → a vision encoder stage feeding the decoder (when a matching
/// vision bundle is supplied).
pub fn route(
    decoder: Arc<ModelBundle>,
    vision: Option<Arc<ModelBundle>>,
    modality: Modality,
) -> Pipeline {
    match (modality, vision) {
        (Modality::VisionText, Some(v)) => {
            let mut p = Pipeline::default();
            p.push(Stage {
                bundle: v,
                phase: Phase::Prefill,
                label: "vision_enc".into(),
                device: None,
            });
            p.push(Stage {
                bundle: decoder,
                phase: Phase::Prefill,
                label: "decode".into(),
                device: None,
            });
            p
        }
        _ => Pipeline::single(decoder, Phase::Decode),
    }
}
