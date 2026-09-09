//! §G `GET /v1/models` — list registered slugs.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::serve::openai::{now_secs, ModelCard, ModelList};
use crate::serve::AppState;

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let data = state
        .registry
        .slugs()
        .into_iter()
        .map(|slug| ModelCard {
            id: slug,
            object: "model",
            // REQUIRED by the schema and previously absent. Process start is
            // the honest value: it is when this server began offering the slug.
            created: now_secs(),
            owned_by: "plow",
        })
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}
