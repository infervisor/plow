//! §G `GET /v1/models` and `GET /v1/models/:id` — the served catalogue.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::serve::openai::{ModelCard, ModelList};
use crate::serve::AppState;

/// One card for `id`, whose weights and engine live under `canonical`.
///
/// `created` is the PROCESS START, stamped once into `AppState`: it used to be
/// `now_secs()` evaluated per card, so the field changed on every scrape and a
/// client diffing the catalogue saw every model as new each time.
///
/// For an ALIAS the two differ, and every property is read from `canonical` —
/// an alias has no engine, no residency and no context length of its own.
/// `root`/`parent` name what it resolves to, which is what those fields are for
/// and how a router discovers the relationship.
fn card(state: &AppState, id: String, canonical: &str) -> ModelCard {
    let is_alias = id != canonical;
    let x_plow_endpoints = endpoints(state, canonical);
    let x_plow_modalities = if x_plow_endpoints.contains(&"chat/completions") {
        let media = state
            .registry
            .get(canonical)
            .ok()
            .filter(|_| state.has_gpu_engine(canonical))
            .and_then(|b| b.serve().multimodal.as_ref().map(|c| c.modalities.iter().map(|m| m.kind.clone()).collect::<Vec<_>>()))
            .unwrap_or_default();
        std::iter::once("text".to_string()).chain(media).collect()
    } else {
        Vec::new()
    };
    ModelCard {
        x_plow_endpoints,
        x_plow_modalities,
        max_model_len: state.max_ctx(canonical),
        root: canonical.to_string(),
        parent: is_alias.then(|| canonical.to_string()),
        permission: Vec::new(),
        x_plow_sampling: (!state.sampling_honoured(canonical)).then_some("device_argmax"),
        id,
        object: "model",
        created: state.started(),
        owned_by: "plow",
    }
}

/// The routes `canonical` serves: its card's `x_plow_endpoints`. Exact for a bound (resident)
/// model; a managed model that is not resident reads as a text model until its switch binds it.
pub(crate) fn endpoints(state: &AppState, canonical: &str) -> Vec<&'static str> {
    #[cfg(feature = "cuda")]
    if crate::tts::serving::serves_speech(state, canonical) {
        // A speech LM's tokens are codec / speech ids: it serves audio, not text.
        return vec!["audio/speech"];
    } else if crate::asr::serving::serves_audio(state, canonical)
        || crate::asr::serving::packet_model_names().iter().any(|n| n == canonical)
    {
        // An audio LM's prompt contract needs audio: transcription only, as a packet ASR card.
        return vec!["audio/transcriptions", "audio/transcriptions/stream"];
    }
    let _ = (state, canonical);
    vec!["chat/completions", "completions"]
}

/// 400 `model_not_supported_for_endpoint` for a route `model` does not serve. A text prompt on
/// a speech or audio LM runs its graph on ids it was never built for (a device fault that takes
/// every co-resident engine with it), so the route is refused before anything is submitted.
pub(crate) fn refuse_unserved(state: &AppState, model: &str, endpoint: &str) -> Option<Response> {
    let served = endpoints(state, model);
    (!served.contains(&endpoint)).then(|| unserved(model, endpoint, &served))
}

pub(crate) fn unserved(model: &str, endpoint: &str, served: &[&str]) -> Response {
    let served: Vec<String> = served.iter().map(|e| format!("/v1/{e}")).collect();
    crate::serve::api_error(
        axum::http::StatusCode::BAD_REQUEST,
        format!("model '{model}' does not serve /v1/{endpoint}; it serves {}", served.join(", ")),
        "invalid_request_error",
        Some("model_not_supported_for_endpoint"),
        Some("model".into()),
    )
}

/// A packet ASR model hosted beside the registry (`--asr-packet`): transcription only.
#[cfg(feature = "cuda")]
fn audio_card(state: &AppState, id: String) -> ModelCard {
    ModelCard {
        x_plow_endpoints: vec!["audio/transcriptions", "audio/transcriptions/stream"],
        x_plow_modalities: Vec::new(),
        max_model_len: None,
        root: id.clone(),
        parent: None,
        permission: Vec::new(),
        x_plow_sampling: None,
        id,
        object: "model",
        created: state.started(),
        owned_by: "plow",
    }
}

/// Every registered slug, then every alias — a client that hardcodes a served
/// name must be able to SEE it in the catalogue, or it cannot discover that the
/// name works.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let mut data: Vec<ModelCard> = state
        .registry
        .slugs()
        .into_iter()
        .map(|slug| {
            let canonical = slug.clone();
            card(&state, slug, &canonical)
        })
        .collect();
    data.extend(
        state
            .registry
            .alias_pairs()
            .into_iter()
            .map(|(alias, canonical)| card(&state, alias, &canonical)),
    );
    #[cfg(feature = "cuda")]
    data.extend(crate::asr::serving::packet_model_names().into_iter().map(|name| audio_card(&state, name)));
    Json(ModelList {
        object: "list",
        data,
    })
}

/// `GET /v1/models/:id`. Routers fetch a single card to discover
/// `max_model_len` before sizing a request; without this route they got a 404
/// from a server that was serving the model perfectly well.
pub async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    #[cfg(feature = "cuda")]
    if crate::asr::serving::packet_model_names().contains(&id) {
        return Json(audio_card(&state, id)).into_response();
    }
    let canonical = state.registry.resolve(&id).unwrap_or_else(|| id.clone());
    if !state.registry.contains(&canonical) {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{id}'."),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    }
    Json(card(&state, id, &canonical)).into_response()
}

/// Any non-GET method on `/v1/models/:id`.
///
/// A 404, not a 405: this path is a READ of one model card, and the only other
/// things that live under `/v1/models/` are the privileged control routes,
/// which are mounted on a separate listener. Answering 405 there would tell an
/// unprivileged caller that `/v1/models/load` is a real endpoint somewhere.
pub async fn model_route_fallback() -> Response {
    crate::serve::api_error(
        axum::http::StatusCode::NOT_FOUND,
        "no such route",
        "invalid_request_error",
        None,
        None,
    )
}
