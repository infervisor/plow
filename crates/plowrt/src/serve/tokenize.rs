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
    /// Defaults to TRUE, matching vLLM's `/tokenize` and plowrt's own
    /// `/v1/completions`. It defaulted to false here, so the same prompt
    /// tokenized differently depending on which endpoint a client asked —
    /// which is precisely what an alignment endpoint exists to rule out.
    #[serde(default = "default_true")]
    add_special_tokens: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct TokenizeResponse {
    /// vLLM's `/tokenize` shape. `count` was missing, so a client written
    /// against vLLM read `None` for it.
    count: usize,
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
    match state.registry.get(&req.model) {
        Ok(bundle) => {
            let tokens = bundle
                .tokenizer()
                .encode_with_special_tokens(&req.prompt, req.add_special_tokens);
            Json(TokenizeResponse {
                count: tokens.len(),
                tokens,
            })
            .into_response()
        }
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
