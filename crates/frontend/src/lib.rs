//! `frontend` — the Hub layer over `nn-graph`'s model zoo.
//!
//! Network definitions, config parsing, and graph building all live in
//! [`nn_graph::models`]. This crate adds only what genuinely belongs downstream:
//! resolving a HuggingFace model *identifier* and downloading its `config.json`
//! over the network (the `hub` feature). The offline builders are re-exported
//! for convenience.

pub use nn_graph::models::{
    build_from_config_json, build_from_config_json_at, build_graph, BuildError, ModelConfig,
    ShapeBucket,
};

/// Resolve a HuggingFace model identifier (e.g. `"google/gemma-3-4b-it"`) to its
/// `config.json` and build the graph specialized to `bucket`. Requires the
/// `hub` feature.
#[cfg(feature = "hub")]
pub fn build_from_pretrained(
    model_id: &str,
    bucket: &ShapeBucket,
) -> Result<nn_graph::Graph, FrontendError> {
    let json = hub::fetch_config(model_id)?;
    Ok(build_from_config_json_at(&json, bucket)?)
}

#[cfg(feature = "hub")]
#[derive(thiserror::Error, Debug)]
pub enum FrontendError {
    #[error(transparent)]
    Build(#[from] BuildError),
    #[error("hub error: {0}")]
    Hub(String),
}

#[cfg(feature = "hub")]
mod hub {
    use crate::FrontendError;
    use hf_hub::api::sync::Api;

    /// Download `config.json` for a model id and return its contents.
    pub fn fetch_config(model_id: &str) -> Result<String, FrontendError> {
        let api = Api::new().map_err(|e| FrontendError::Hub(e.to_string()))?;
        let path = api
            .model(model_id.to_string())
            .get("config.json")
            .map_err(|e| FrontendError::Hub(e.to_string()))?;
        std::fs::read_to_string(path).map_err(|e| FrontendError::Hub(e.to_string()))
    }
}
