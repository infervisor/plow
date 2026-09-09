//! §I.5 Control plane — load, unload and inspect one model while the server
//! stays up.
//!
//! These are deliberately NOT part of the OpenAI surface. `GET /v1/models`
//! keeps listing every registered slug whatever its residency: a client must
//! not see a model vanish from the catalogue because an operator moved it off
//! a card. Residency lives here instead.
//!
//! **These routes are privileged.** `load` names a directory whose cubins the
//! process will execute, and `unload` terminates other people's requests. The
//! server has no authentication of its own, so the mounting decision is the
//! access control: keep them on the UDS listener (mode 0600) or a loopback
//! admin port, not on the public router.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::serve::{AppState, Residency};

#[derive(Debug, Deserialize)]
pub struct LoadRequest {
    /// Slug to serve the model under.
    pub model: String,
    /// Assets dir to register when the slug is not already known. Must sit
    /// under a `--models-root`.
    #[serde(default)]
    pub assets: Option<String>,
    /// Checkpoint dir; defaults to `<assets>/checkpoint`.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Device ordinal to place a NEW model on. Ignored for a slug that is
    /// already registered — a model does not move between GPUs by being
    /// loaded again; unload it first.
    #[serde(default)]
    pub device: Option<u32>,
    /// Evict LRU co-tenants if the model does not otherwise fit. Off by
    /// default — an admin load must not silently take down another model.
    #[serde(default)]
    pub evict: bool,
}

#[derive(Debug, Deserialize)]
pub struct UnloadRequest {
    pub model: String,
    /// Also drop the registry entry, so the slug leaves `GET /v1/models`.
    /// Default false: the slug stays listed as registered-but-not-resident and
    /// can be loaded again by name alone.
    #[serde(default)]
    pub deregister: bool,
}

#[derive(Debug, Serialize)]
pub struct LoadResponse {
    pub model: String,
    pub state: &'static str,
    pub load_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct UnloadResponse {
    pub model: String,
    pub state: &'static str,
    /// Time spent stopping and flushing in-flight generations.
    pub stop_ms: f64,
    pub unload_ms: f64,
    pub freed_mib: u64,
    pub pool_trimmed_mib: u64,
    pub deregistered: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelStatus {
    pub model: String,
    /// `auto` | `unloading` | `unloaded`.
    pub residency: &'static str,
    /// Whether an engine is installed right now.
    pub resident: bool,
    /// Whether a dispatcher is accepting work.
    pub serving: bool,
    /// Planner requirement in MiB (tensors + measured overhead), when managed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_mib: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_mib: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_mib: Option<u64>,
    /// Device ordinals this model is placed on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<u32>>,
    /// Index of its device group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<usize>,
}

/// One device group: its ordinals, its memory, and what is resident on it.
#[derive(Debug, Serialize)]
pub struct GroupStatus {
    pub group: usize,
    pub devices: Vec<u32>,
    pub free_mib: u64,
    pub total_mib: u64,
    pub resident: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub models: Vec<ModelStatus>,
    /// One entry per device group. Empty on a CPU-only serve.
    pub groups: Vec<GroupStatus>,
}

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (status, Json(serde_json::json!({ "error": msg.to_string() }))).into_response()
}

#[cfg(feature = "cuda")]
const MIB: u64 = 1 << 20;

/// Resolve a requested assets dir against the `--models-root` allow-list.
///
/// Both sides are canonicalized before the prefix test, so `..` segments and
/// symlinks cannot walk out of a root. An empty allow-list refuses everything
/// rather than defaulting open.
fn resolve_assets(state: &AppState, dir: &str) -> std::result::Result<PathBuf, Response> {
    let roots = state.models_roots();
    if roots.is_empty() {
        return Err(err(
            StatusCode::FORBIDDEN,
            "no --models-root is configured, so no assets dir may be loaded by request",
        ));
    }
    let canon = PathBuf::from(dir).canonicalize().map_err(|e| {
        err(
            StatusCode::BAD_REQUEST,
            format_args!("assets dir {dir}: {e}"),
        )
    })?;
    if !roots.iter().any(|r| canon.starts_with(r)) {
        return Err(err(
            StatusCode::FORBIDDEN,
            format_args!(
                "{} is not under any --models-root ({})",
                canon.display(),
                roots
                    .iter()
                    .map(|r| r.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    Ok(canon)
}

/// `POST /v1/models/load` — make a model resident, registering its assets dir
/// first when the slug is new.
pub async fn load(State(state): State<Arc<AppState>>, Json(req): Json<LoadRequest>) -> Response {
    // The path check comes FIRST, before any backend or slug lookup. It is a
    // security boundary, and a boundary that only fires on builds which happen
    // to have a GPU manager is not one — the refusal must not depend on how far
    // the request would otherwise have got.
    let assets = match req.assets.as_deref().map(|d| resolve_assets(&state, d)) {
        Some(Err(refusal)) => return refusal,
        Some(Ok(dir)) => Some(dir),
        None => None,
    };

    #[cfg(feature = "cuda")]
    {
        if state.managers().is_empty() {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                "no GPU model manager on this server",
            );
        }

        // An already-registered slug keeps its group: loading is not a way to
        // migrate a model between GPUs, and silently moving one would strand
        // whatever capacity the operator had planned around it.
        let target = match state.manager_for(&req.model) {
            Some(m) => Some((state.slug_group(&req.model).unwrap_or(0), m.clone())),
            None => {
                let group = match req.device {
                    Some(d) => match state
                        .managers()
                        .iter()
                        .position(|m| m.ordinals().first() == Some(&d))
                    {
                        Some(g) => Some(g),
                        None => {
                            return err(
                                StatusCode::BAD_REQUEST,
                                format_args!("no device group starts at ordinal {d}"),
                            )
                        }
                    },
                    None => None,
                };
                state
                    .manager_for_new(group)
                    .map(|(g, m)| (g, m.clone()))
            }
        };
        let Some((group, mgr)) = target else {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                "no GPU model manager on this server",
            );
        };

        if let Some(dir) = assets {
            // Registering the bundle is what makes the tokenizer, chat template
            // and bucket ladder reachable; registering with the manager is what
            // gives it a VRAM plan. A slug already known to either is left alone
            // so the call stays idempotent.
            if !state.registry.contains(&req.model) {
                if let Err(e) = state.registry.load(&dir, Some(req.model.clone())) {
                    return err(StatusCode::BAD_REQUEST, e);
                }
            }
            let ckpt = req
                .checkpoint
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| dir.join("checkpoint"));
            if let Err(e) = mgr.register(&req.model, dir, ckpt) {
                return err(StatusCode::BAD_REQUEST, e);
            }
            state.set_slug_group(&req.model, group);
        }

        if !mgr.manages(&req.model) {
            return err(
                StatusCode::NOT_FOUND,
                format_args!(
                    "unknown model {:?} — pass \"assets\" to register a new one",
                    req.model
                ),
            );
        }

        use crate::serve::manager::EnsureError;
        return match mgr.load(&req.model, req.evict).await {
            Ok(load_ms) => Json(LoadResponse {
                model: req.model,
                state: "resident",
                load_ms,
            })
            .into_response(),
            // 409, not 503: the request is answerable, it just conflicts with
            // what is already resident. Retrying unchanged will not help;
            // asking for eviction, or unloading something, will.
            Err(e @ EnsureError::WontFit { .. }) => err(StatusCode::CONFLICT, e),
            Err(e @ EnsureError::Unloaded) => err(StatusCode::CONFLICT, e),
            Err(EnsureError::Load(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (state, req, assets);
        err(
            StatusCode::NOT_IMPLEMENTED,
            "model load/unload needs a GPU backend",
        )
    }
}

/// `POST /v1/models/unload` — stop and flush this model, release its device
/// memory, and leave every other model serving.
pub async fn unload(State(state): State<Arc<AppState>>, Json(req): Json<UnloadRequest>) -> Response {
    #[cfg(feature = "cuda")]
    {
        let Some(mgr) = state.manager_for(&req.model).cloned() else {
            return err(
                StatusCode::NOT_FOUND,
                format_args!("no model manager serves {:?}", req.model),
            );
        };
        if !mgr.manages(&req.model) {
            return err(
                StatusCode::NOT_FOUND,
                format_args!("unknown model {:?}", req.model),
            );
        }
        use crate::serve::manager::EnsureError;
        let report = match mgr.unload(&req.model).await {
            Ok(r) => r,
            Err(EnsureError::Load(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
        if req.deregister {
            let _ = state.registry.unload(&req.model);
        }
        return Json(UnloadResponse {
            model: req.model,
            state: "unloaded",
            stop_ms: report.stop_ms,
            unload_ms: report.unload_ms,
            freed_mib: report.freed / MIB,
            pool_trimmed_mib: report.pool_trimmed / MIB,
            deregistered: req.deregister,
        })
        .into_response();
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (state, req);
        err(
            StatusCode::NOT_IMPLEMENTED,
            "model load/unload needs a GPU backend",
        )
    }
}

/// `GET /v1/models/status` — residency and footprint per registered slug.
pub async fn status(State(state): State<Arc<AppState>>) -> Response {
    let mut models = Vec::new();
    for slug in state.registry.slugs() {
        let residency = match state.residency(&slug) {
            Residency::Auto => "auto",
            Residency::Unloading => "unloading",
            Residency::Unloaded => "unloaded",
        };
        #[allow(unused_mut)]
        let mut entry = ModelStatus {
            resident: state.has_gpu_engine(&slug),
            serving: state.mux(&slug).is_some(),
            residency,
            required_mib: None,
            weights_mib: None,
            kv_mib: None,
            devices: None,
            group: None,
            model: slug.clone(),
        };
        #[cfg(feature = "cuda")]
        if let Some(mgr) = state.manager_for(&slug) {
            entry.required_mib = mgr.required(&slug).map(|b| b / MIB);
            if let Some(plan) = mgr.plan(&slug) {
                entry.weights_mib = Some(plan.weights_bytes / MIB);
                entry.kv_mib = Some(plan.kv_bytes / MIB);
            }
            entry.devices = Some(mgr.ordinals());
            entry.group = state.slug_group(&slug);
        }
        models.push(entry);
    }

    #[allow(unused_mut)]
    let mut groups: Vec<GroupStatus> = Vec::new();
    #[cfg(feature = "cuda")]
    for (i, mgr) in state.managers().iter().enumerate() {
        let (free, total) = mgr.device_mem_info().unwrap_or((0, 0));
        groups.push(GroupStatus {
            group: i,
            devices: mgr.ordinals(),
            free_mib: free / MIB,
            total_mib: total / MIB,
            resident: mgr
                .slugs()
                .into_iter()
                .filter(|s| state.has_gpu_engine(s))
                .collect(),
        });
    }

    Json(StatusResponse { models, groups }).into_response()
}
