//! §G `GET /v1/models` — list registered slugs.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::serve::openai::{ModelCard, ModelList};
use crate::serve::AppState;

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let data = state
        .registry
        .slugs()
        .map(|slug| ModelCard {
            id: slug.to_string(),
            object: "model",
            owned_by: "plow",
        })
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}
