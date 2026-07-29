//! Resolve a HuggingFace model *identifier* to its `config.json` and build the
//! graph.
//!
//! This was its own crate (`frontend`), for one reason: `nn-graph` was an
//! external git dependency, so anything that wanted to sit *downstream* of the
//! model zoo needed somewhere else to live. Vendoring the crate removed that
//! constraint, and what was left was 12 lines of re-export plus the 10-line
//! `fetch_config` below — a compilation unit, a `Cargo.toml` and a workspace
//! entry to wrap one network call. It lives next to the models it resolves now.
//!
//! Still feature-gated (`hub`): the offline path — every `--hf-dir` compile, and
//! all of `costmodel` — must not drag in a TLS stack to look at a `DType`.

use crate::models::{build_from_config_json_at, BuildError, ShapeBucket};
use crate::Graph;

#[derive(thiserror::Error, Debug)]
pub enum HubError {
    #[error(transparent)]
    Build(#[from] BuildError),
    #[error("hub error: {0}")]
    Hub(String),
}

/// Resolve a HuggingFace model identifier (e.g. `"google/gemma-3-4b-it"`) to its
/// `config.json` and build the graph specialized to `bucket`.
pub fn build_from_pretrained(model_id: &str, bucket: &ShapeBucket) -> Result<Graph, HubError> {
    let json = fetch_config(model_id)?;
    Ok(build_from_config_json_at(&json, bucket)?)
}

/// Download `config.json` for a model id and return its contents.
pub fn fetch_config(model_id: &str) -> Result<String, HubError> {
    use hf_hub::api::sync::Api;

    let api = Api::new().map_err(|e| HubError::Hub(e.to_string()))?;
    let path = api
        .model(model_id.to_string())
        .get("config.json")
        .map_err(|e| HubError::Hub(e.to_string()))?;
    std::fs::read_to_string(path).map_err(|e| HubError::Hub(e.to_string()))
}
