//! Tokenizer alignment endpoints used by serving benchmark clients.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::serve::AppState;

#[derive(Deserialize)]
pub struct TokenizeRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    add_special_tokens: bool,
}

#[derive(Serialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    model: String,
    tokens: Vec<u32>,
}

#[derive(Serialize)]
struct DetokenizeResponse {
    prompt: String,
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    if req.add_special_tokens {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "add_special_tokens=true is not supported"})),
        )
            .into_response();
    }
    match state.registry.get(&req.model) {
        Ok(bundle) => Json(TokenizeResponse {
            tokens: bundle.tokenizer().encode(&req.prompt),
        })
        .into_response(),
        Err(_) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("no model registered for '{}'.", req.model)})),
        )
            .into_response(),
    }
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    match state.registry.get(&req.model) {
        Ok(bundle) => Json(DetokenizeResponse {
            prompt: bundle.tokenizer().decode(&req.tokens),
        })
        .into_response(),
        Err(_) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("no model registered for '{}'.", req.model)})),
        )
            .into_response(),
    }
}
