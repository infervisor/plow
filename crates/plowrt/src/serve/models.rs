//! §G `GET /v1/models` and `GET /v1/models/{id}` — the served catalogue.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::serve::openai::{ModelCard, ModelList};
use crate::serve::AppState;

/// One card. `created` is the PROCESS START, stamped once into `AppState`:
/// it used to be `now_secs()` evaluated per card, so the field changed on every
/// scrape and a client diffing the catalogue saw every model as new each time.
fn card(state: &AppState, slug: String) -> ModelCard {
    ModelCard {
        max_model_len: state.max_ctx(&slug),
        root: slug.clone(),
        parent: None,
        permission: Vec::new(),
        x_plow_sampling: (!state.sampling_honoured(&slug)).then_some("device_argmax"),
        id: slug,
        object: "model",
        created: state.started(),
        owned_by: "plow",
    }
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let data = state
        .registry
        .slugs()
        .into_iter()
        .map(|slug| card(&state, slug))
        .collect();
    Json(ModelList {
        object: "list",
        data,
    })
}

/// `GET /v1/models/{id}`. Routers fetch a single card to discover
/// `max_model_len` before sizing a request; without this route they got a 404
/// from a server that was serving the model perfectly well.
pub async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if !state.registry.slugs().iter().any(|s| *s == id) {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{id}'."),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    }
    Json(card(&state, id)).into_response()
}
