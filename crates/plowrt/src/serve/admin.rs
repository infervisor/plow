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

use crate::serve::AppState;

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
    /// Extra names this model answers to (vLLM's `--served-model-name`).
    /// Registered after the model itself, so a rejected alias never leaves a
    /// half-loaded model behind.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// A data-parallel model's rank to load (default: every rank). `device` also selects the
    /// rank placed there.
    #[serde(default)]
    pub rank: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct UnloadRequest {
    pub model: String,
    /// Also drop the registry entry, so the slug leaves `GET /v1/models`.
    /// Default false: the slug stays listed as registered-but-not-resident and
    /// can be loaded again by name alone.
    #[serde(default)]
    pub deregister: bool,
    /// A data-parallel model's rank to unload (default: every rank). Its in-flight work drains
    /// (bounded by `PLOW_DRAIN_TIMEOUT_MS`, 30 s unset) while new work routes to the others.
    #[serde(default)]
    pub rank: Option<usize>,
    /// Select the rank by the device its group starts at.
    #[serde(default)]
    pub device: Option<u32>,
    /// Drain timeout in milliseconds for in-flight requests. `Some(0)` preempts immediately;
    /// unset defaults to bounded drain via `PLOW_DRAIN_TIMEOUT_MS` (or 30 s).
    #[serde(default)]
    pub drain_timeout_ms: Option<u64>,
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
    /// Data-parallel ranks, when the model has more than one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp: Option<Vec<RankStatus>>,
}

/// One DP rank of a model.
#[derive(Debug, Serialize)]
pub struct RankStatus {
    pub rank: usize,
    /// Its instance key (`model#rank`); also accepted as `model` by load/unload.
    pub key: String,
    pub device: u32,
    pub group: usize,
    pub residency: &'static str,
    pub resident: bool,
    pub serving: bool,
    pub queued: u64,
    pub active: u64,
    pub capacity: u64,
    pub load: f32,
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
    pub capabilities: ControlCapabilities,
    pub models: Vec<ModelStatus>,
    /// One entry per device group. Empty on a CPU-only serve.
    pub groups: Vec<GroupStatus>,
}

#[derive(Debug, Serialize)]
pub struct ControlCapabilities {
    pub load: bool,
    pub unload: bool,
    pub register: bool,
    pub eviction: bool,
}

pub(super) fn err(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": msg.to_string() })),
    )
        .into_response()
}

const MIB: u64 = 1 << 20;

/// Resolve a requested assets dir against the `--models-root` allow-list.
///
/// Both sides are canonicalized before the prefix test, so `..` segments and
/// symlinks cannot walk out of a root. An empty allow-list refuses everything
/// rather than defaulting open.
fn resolve_assets(state: &AppState, dir: &str) -> std::result::Result<PathBuf, Response> {
    if state.models_roots().is_empty() {
        return Err(no_roots());
    }
    let canon = PathBuf::from(dir).canonicalize().map_err(|e| {
        err(
            StatusCode::BAD_REQUEST,
            format_args!("assets dir {dir}: {e}"),
        )
    })?;
    authorize(state, canon)
}

/// The allow-list a serve starts with: every `--models-root`, the parents of the assets dirs
/// it was started with (two bundles side by side need no flag), and the local store's
/// `bundles/`, where signature-checked pulls land. Never the working directory: started in `/`
/// (systemd's default) that made every directory on the host loadable.
pub fn startup_models_roots(
    explicit: &[String],
    assets: &[PathBuf],
    store_bundles: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = explicit.iter().map(PathBuf::from).collect();
    roots.extend(assets.iter().filter_map(|a| a.parent().map(PathBuf::from)));
    roots.extend(store_bundles);
    roots.dedup();
    roots
}

fn no_roots() -> Response {
    err(
        StatusCode::FORBIDDEN,
        "no --models-root is configured, so no assets dir may be loaded by request",
    )
}

/// The one allow-list test every assets dir a load names or finds must pass: `canon` (already
/// canonical) sits under a `--models-root`.
fn authorize(state: &AppState, canon: PathBuf) -> std::result::Result<PathBuf, Response> {
    let roots = state.models_roots();
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

fn find_local_bundle(
    state: &AppState,
    model: &str,
    req_checkpoint: Option<&str>,
) -> Option<(PathBuf, PathBuf)> {
    let has_model_files = |dir: &std::path::Path| {
        dir.is_dir()
            && (dir.join("manifest.toml").is_file()
                || dir.join("weights.json").is_file()
                || dir.join("config.json").is_file()
                || dir.join("s3gen.pkt").is_file()
                || dir.join("codec.pkt").is_file())
    };

    // A NAME under a root, never a path: an absolute `model` would replace the root in `join`,
    // and the canonical result must still sit under a root (no `..` or symlink out).
    let roots = state.models_roots();
    let model_path = std::path::Path::new(model);
    if model_path.is_relative() {
        for root in roots {
            let Ok(canon) = root.join(model_path).canonicalize() else {
                continue;
            };
            if roots.iter().any(|r| canon.starts_with(r)) && has_model_files(&canon) {
                let ckpt = req_checkpoint
                    .map(PathBuf::from)
                    .unwrap_or_else(|| crate::asset::serve::checkpoint_dir(&canon));
                return Some((canon, ckpt));
            }
        }
    }

    if let Ok(st) = crate::dist::Store::open_default() {
        if let Ok(r) = crate::dist::reference::parse(model) {
            if let Some(variant_id) = st.pinned(&r.pin_key()) {
                let bdir = st.root().join("bundles").join(&variant_id);
                if has_model_files(&bdir) {
                    let ckpt = req_checkpoint
                        .map(PathBuf::from)
                        .unwrap_or_else(|| crate::asset::serve::checkpoint_dir(&bdir));
                    return Some((bdir, ckpt));
                }
            }
        }
        for p in st.list() {
            if p.reference == model || p.reference.ends_with(&format!("/{}", model)) {
                if has_model_files(&p.dir) {
                    let ckpt = req_checkpoint
                        .map(PathBuf::from)
                        .unwrap_or_else(|| crate::asset::serve::checkpoint_dir(&p.dir));
                    return Some((p.dir, ckpt));
                }
            }
        }
    }

    None
}

#[cfg(feature = "dist")]
async fn fetch_from_pdist_and_hf(
    state: &AppState,
    model: &str,
    req_checkpoint: Option<&str>,
) -> std::result::Result<(PathBuf, PathBuf), String> {
    let model_str = model.to_string();
    let req_ckpt = req_checkpoint.map(|s| s.to_string());
    let backend = state.execset.backend().clone();

    tokio::task::spawn_blocking(move || -> std::result::Result<(PathBuf, PathBuf), String> {
        let mut r = crate::dist::reference::parse(&model_str)
            .map_err(|e| format!("parse model reference {model_str:?}: {e}"))?;
        if r.registry == crate::dist::reference::DEFAULT_REGISTRY {
            r.registry = crate::config::RuntimeConfig::get().registry.clone();
        }

        let f = crate::dist::transport(&r.registry)
            .map_err(|e| format!("connect to registry {}: {e}", r.registry))?;
        let st = crate::dist::Store::open_default()
            .map_err(|e| format!("open store: {e}"))?;
        let _lock = st.lock().map_err(|e| format!("lock store: {e}"))?;

        let live = crate::dist::live_target(&[backend]);
        let trust = crate::dist::sign::Trust::for_registry(
            crate::config::RuntimeConfig::signing_env().1.as_deref(),
            &r.registry,
        ).map_err(|e| format!("verify trust for {}: {e}", r.registry))?;

        let c = plow_asset::dist::Constraints::default();
        let resolved = crate::dist::pull::resolve_trusted(&*f, &r, &live, c, &trust)
            .map_err(|e| format!("resolve variant for {model_str} from {}: {e}", r.registry))?;

        let (dir, t) = crate::dist::pull::pull(&st, &*f, &resolved)
            .map_err(|e| format!("pull bundle for {model_str}: {e}"))?;
        tracing::info!(model = %model_str, transfer = %t, dir = %dir.display(), "pulled bundle from pdist");

        let dist_snapshot = (resolved.bundle.checkpoint.source == plow_asset::dist::DIST_SOURCE)
            .then(|| crate::dist::weights::dist_snapshot_path(st.root(), &resolved.bundle))
            .transpose()
            .map_err(|e| e.to_string())?;

        let ckpt = req_ckpt
            .map(PathBuf::from)
            .or_else(|| crate::config::RuntimeConfig::get().checkpoint.clone().map(PathBuf::from))
            .or(dist_snapshot)
            .unwrap_or_else(|| crate::dist::prepare::farm_path(st.root(), &resolved.bundle));

        let ckpt = if ckpt.is_dir() {
            ckpt
        } else if resolved.bundle.checkpoint.source.starts_with("hf:") {
            let c = &resolved.bundle.checkpoint;
            tracing::info!(
                source = %c.source,
                revision = %c.revision,
                shards = c.shards,
                "fetching checkpoint weights from Hugging Face"
            );
            crate::dist::weights::fetch(st.root(), &c.source, &c.revision, c.shards)
                .map_err(|e| format!("fetch Hugging Face weights {}: {e}", c.source))?
        } else {
            return Err(format!(
                "{} needs checkpoint {}, but it is not available locally or on Hugging Face",
                model_str, resolved.bundle.checkpoint.source
            ));
        };

        let farm = crate::dist::prepare::build(st.root(), &resolved.bundle, &dir, &ckpt)
            .map_err(|e| format!("build checkpoint farm for {model_str}: {e}"))?;

        let _ = st.pin(&r.pin_key(), &resolved.variant.variant_id);
        tracing::info!(
            model = %model_str,
            bundle = %dir.display(),
            checkpoint = %farm.dir.display(),
            shards = farm.shards,
            "prepared model bundle and checkpoint"
        );
        Ok((dir, farm.dir))
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {e}"))?
}

#[cfg(not(feature = "dist"))]
async fn fetch_from_pdist_and_hf(
    _state: &AppState,
    model: &str,
    _req_checkpoint: Option<&str>,
) -> std::result::Result<(PathBuf, PathBuf), String> {
    Err(format!(
        "model {model:?} is not available locally; downloading from pdist and Hugging Face requires plowrt built with --features dist"
    ))
}

/// A bundle for an unregistered `model` named without `assets`: a local one, else a pdist pull.
/// Either way the directory passes [`authorize`] — a path in `model` (absolute, relative to the
/// working directory, or `..` out of a root) is held to the same allow-list as `assets`, and a pull
/// is refused before any download when the store it lands in is not under a root.
async fn resolve_or_fetch_assets(
    state: &AppState,
    model: &str,
    req_checkpoint: Option<&str>,
) -> std::result::Result<(PathBuf, PathBuf), Response> {
    if state.models_roots().is_empty() {
        return Err(no_roots());
    }
    if let Some((dir, ckpt)) = find_local_bundle(state, model, req_checkpoint) {
        return Ok((authorize(state, dir)?, ckpt));
    }
    if let Ok(st) = crate::dist::Store::open_default() {
        let bundles = st.root().join("bundles");
        let bundles = bundles.canonicalize().unwrap_or(bundles);
        authorize(state, bundles)?;
    }
    let (dir, ckpt) = fetch_from_pdist_and_hf(state, model, req_checkpoint)
        .await
        .map_err(|e| {
            err(
                StatusCode::NOT_FOUND,
                format_args!(
                    "model {model:?} not registered, not found locally, and download failed: {e}"
                ),
            )
        })?;
    let dir = dir.canonicalize().map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format_args!("{}: {e}", dir.display()),
        )
    })?;
    Ok((authorize(state, dir)?, ckpt))
}

/// Undo a registration this load made: the manager plan, registry entry (and its aliases),
/// group and checkpoint pairing.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
async fn roll_back(
    state: &AppState,
    mgr: &crate::serve::manager::ModelManager,
    slug: &str,
    dir: &std::path::Path,
) {
    if let Err(e) = mgr.unload_and_deregister(slug, true).await {
        tracing::warn!(%slug, error = %e, "load rollback: deregister failed");
    }
    forget_checkpoint(state, dir);
}

/// Drop `dir`'s checkpoint pairing unless another registered slug still serves it.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
fn forget_checkpoint(state: &AppState, dir: &std::path::Path) {
    let shared = state.registry.slugs().iter().any(|s| {
        state
            .registry
            .get(s)
            .is_ok_and(|b| b.dir.canonicalize().ok().as_deref() == Some(dir))
    });
    if !shared {
        crate::asset::serve::clear_checkpoint(dir);
    }
}

/// `POST /v1/models/load` — make a model resident, registering its assets dir
/// first when the slug is new.
pub async fn load(State(state): State<Arc<AppState>>, Json(req): Json<LoadRequest>) -> Response {
    match tokio::spawn(load_inner(state, req)).await {
        Ok(response) => response,
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
async fn load_inner(state: Arc<AppState>, mut req: LoadRequest) -> Response {
    req.model = state.registry.resolve(&req.model).unwrap_or(req.model);
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    if let Some(keys) = dp_targets(&state, &req.model, req.rank, req.device) {
        return match keys {
            Ok(keys) => dp_load(&state, keys, req.evict).await,
            Err(refusal) => refusal,
        };
    }
    let _control = state.control_lock(&req.model).await;
    // The path check comes FIRST, before any backend or slug lookup. It is a
    // security boundary, and a boundary that only fires on builds which happen
    // to have a GPU manager is not one — the refusal must not depend on how far
    // the request would otherwise have got.
    //
    // A checkpoint override names a directory the load reads (weights, tokenizer, templates):
    // held to the same allow-list as the assets dir.
    if let Some(c) = req.checkpoint.take() {
        match resolve_assets(&state, &c) {
            Ok(dir) => req.checkpoint = Some(dir.display().to_string()),
            Err(refusal) => return refusal,
        }
    }
    let (assets, checkpoint_override) = if let Some(d) = req.assets.as_deref() {
        match resolve_assets(&state, d) {
            Ok(dir) => (Some(dir), req.checkpoint.as_deref().map(PathBuf::from)),
            Err(refusal) => return refusal,
        }
    } else if !state.registry.contains(&req.model) {
        match resolve_or_fetch_assets(&state, &req.model, req.checkpoint.as_deref()).await {
            Ok((dir, ckpt)) => (Some(dir), Some(ckpt)),
            Err(refusal) => return refusal,
        }
    } else {
        (None, req.checkpoint.as_deref().map(PathBuf::from))
    };

    #[cfg(feature = "cpu")]
    if let Some(manager) = state.portable_manager() {
        if let Some(ref ckpt) = checkpoint_override {
            req.checkpoint = Some(ckpt.display().to_string());
        }
        return manager.load(&state, req, assets).await;
    }

    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    {
        if state.managers().is_empty() {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                "no model manager on this server",
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
                state.manager_for_new(group).map(|(g, m)| (g, m.clone()))
            }
        };
        let Some((group, mgr)) = target else {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                "no model manager on this server",
            );
        };

        let mut registered_dir: Option<PathBuf> = None;
        if let Some(dir) = assets {
            // Registering the bundle is what makes the tokenizer, chat template
            // and bucket ladder reachable; registering with the manager is what
            // gives it a VRAM plan. A slug already known to either is left alone
            // so the call stays idempotent.
            let newly_registered = !state.registry.contains(&req.model);
            if let Ok(existing) = state.registry.get(&req.model) {
                if existing.dir.canonicalize().ok().as_ref() != Some(&dir) {
                    return err(
                        StatusCode::CONFLICT,
                        "model is registered from different assets; deregister it first",
                    );
                }
            }
            let ckpt = checkpoint_override
                .or_else(|| req.checkpoint.as_ref().map(PathBuf::from))
                .unwrap_or_else(|| crate::asset::serve::checkpoint_dir(&dir));
            if newly_registered {
                // The pairing is recorded first so the bundle's pin check and every HF-file read
                // at load see the requested checkpoint.
                if let Err(e) = crate::asset::serve::set_checkpoint(&dir, &ckpt) {
                    return err(StatusCode::CONFLICT, e);
                }
                if let Err(e) = state.registry.load(&dir, Some(req.model.clone())) {
                    forget_checkpoint(&state, &dir);
                    return err(StatusCode::BAD_REQUEST, e);
                }
            }
            if let Err(e) = mgr.register(&req.model, dir.clone(), ckpt) {
                if newly_registered {
                    let _ = state.registry.unload(&req.model);
                    forget_checkpoint(&state, &dir);
                }
                return err(StatusCode::BAD_REQUEST, e);
            }
            state.set_slug_group(&req.model, group);
            for alias in &req.aliases {
                if let Err(e) = state.registry.add_alias(alias.clone(), &req.model) {
                    if newly_registered {
                        roll_back(&state, &mgr, &req.model, &dir).await;
                    }
                    return err(StatusCode::BAD_REQUEST, e);
                }
            }
            if newly_registered {
                registered_dir = Some(dir);
            }
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
        let loaded = mgr.load(&req.model, req.evict).await;
        // A model this call registered and could not load leaves no trace: kept, it stayed in
        // the catalogue and every request for it paid another full load attempt.
        if let (Err(_), Some(dir)) = (&loaded, &registered_dir) {
            roll_back(&state, &mgr, &req.model, dir).await;
        }
        match loaded {
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
            Err(e @ EnsureError::SwitchTimeout(_)) => err(StatusCode::SERVICE_UNAVAILABLE, e),
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
    {
        let _ = (state, req, assets, checkpoint_override);
        err(
            StatusCode::NOT_IMPLEMENTED,
            "no model manager on this server",
        )
    }
}

/// `POST /v1/models/unload` — stop and flush this model, release its device
/// memory, and leave every other model serving.
pub async fn unload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnloadRequest>,
) -> Response {
    match tokio::spawn(unload_inner(state, req)).await {
        Ok(response) => response,
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
async fn unload_inner(state: Arc<AppState>, mut req: UnloadRequest) -> Response {
    req.model = state.registry.resolve(&req.model).unwrap_or(req.model);
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    if let Some(keys) = dp_targets(&state, &req.model, req.rank, req.device) {
        if req.deregister {
            return err(
                StatusCode::BAD_REQUEST,
                "a data-parallel model cannot be deregistered while serving",
            );
        }
        return match keys {
            Ok(keys) => dp_unload(&state, keys, req.drain_timeout_ms).await,
            Err(refusal) => refusal,
        };
    }
    let _control = state.control_lock(&req.model).await;
    #[cfg(feature = "cpu")]
    if let Some(manager) = state.portable_manager() {
        return manager.unload(&state, req).await;
    }
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
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
        let report = match mgr
            .unload_with_drain(
                &req.model,
                req.deregister,
                unload_drain(req.drain_timeout_ms),
            )
            .await
        {
            Ok(r) => r,
            Err(EnsureError::Load(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
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
    #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
    {
        let _ = state;
        err(
            StatusCode::NOT_FOUND,
            format_args!("unknown model {:?}", req.model),
        )
    }
}

/// How long an unload lets in-flight generations finish before preempting the rest: the
/// request's `drain_timeout_ms`, else `PLOW_DRAIN_TIMEOUT_MS`, else 30 s. `None` (a 0) preempts
/// at once. One rule for every backend: ordinary GPU unloads used to preempt at once and the
/// CPU/Metal ones to drain unbounded, against the documented 30 s.
pub fn unload_drain(drain_timeout_ms: Option<u64>) -> Option<std::time::Duration> {
    let ms = drain_timeout_ms
        .or_else(|| crate::config::RuntimeConfig::get().drain_timeout_ms())
        .unwrap_or(30_000);
    (ms > 0).then(|| std::time::Duration::from_millis(ms))
}

/// The instance keys a load/unload of `model` addresses, when it is a DP model or one of its
/// ranks: `None` for an ordinary model.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
fn dp_targets(
    state: &AppState,
    model: &str,
    rank: Option<usize>,
    device: Option<u32>,
) -> Option<std::result::Result<Vec<String>, Response>> {
    if let Some((set, r)) = state.dp_rank(model) {
        return Some(Ok(vec![set.ranks[r].key.clone()]));
    }
    let set = state.dp_set(model)?;
    let keys: Vec<String> = set
        .ranks
        .iter()
        .filter(|r| rank.is_none_or(|n| r.rank == n) && device.is_none_or(|d| r.ordinal == d))
        .map(|r| r.key.clone())
        .collect();
    Some(if keys.is_empty() {
        Err(err(
            StatusCode::NOT_FOUND,
            format_args!("{model} has no DP rank matching rank {rank:?} device {device:?}"),
        ))
    } else {
        Ok(keys)
    })
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
async fn dp_load(state: &Arc<AppState>, keys: Vec<String>, evict: bool) -> Response {
    use crate::serve::manager::EnsureError;
    let mut load_ms = 0f64;
    for key in &keys {
        let _control = state.control_lock(key).await;
        let Some(mgr) = state.manager_for(key).cloned() else {
            return err(
                StatusCode::NOT_FOUND,
                format_args!("no model manager serves {key:?}"),
            );
        };
        match mgr.load(key, evict).await {
            Ok(ms) => load_ms = load_ms.max(ms),
            Err(e @ (EnsureError::WontFit { .. } | EnsureError::Unloaded)) => {
                return err(StatusCode::CONFLICT, format_args!("{key}: {e}"))
            }
            Err(e @ EnsureError::SwitchTimeout(_)) => {
                return err(StatusCode::SERVICE_UNAVAILABLE, format_args!("{key}: {e}"))
            }
            Err(EnsureError::Load(e)) => {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format_args!("{key}: {e}"),
                )
            }
        }
    }
    Json(LoadResponse {
        model: keys.join(","),
        state: "resident",
        load_ms,
    })
    .into_response()
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
async fn dp_unload(
    state: &Arc<AppState>,
    keys: Vec<String>,
    drain_timeout_ms: Option<u64>,
) -> Response {
    let bound = unload_drain(drain_timeout_ms).unwrap_or_default();
    let mut total = UnloadResponse {
        model: keys.join(","),
        state: "unloaded",
        stop_ms: 0.0,
        unload_ms: 0.0,
        freed_mib: 0,
        pool_trimmed_mib: 0,
        deregistered: false,
    };
    let tasks: Vec<_> = keys
        .into_iter()
        .map(|key| {
            let state = Arc::clone(state);
            async move {
                let _control = state.control_lock(&key).await;
                let Some(mgr) = state.manager_for(&key).cloned() else {
                    return Err(err(
                        StatusCode::NOT_FOUND,
                        format_args!("no model manager serves {key:?}"),
                    ));
                };
                mgr.unload_rank(&key, bound).await.map_err(|e| {
                    err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format_args!("{key}: {e}"),
                    )
                })
            }
        })
        .collect();

    for res in futures::future::join_all(tasks).await {
        match res {
            Ok(r) => {
                total.stop_ms = total.stop_ms.max(r.stop_ms);
                total.unload_ms = total.unload_ms.max(r.unload_ms);
                total.freed_mib += r.freed / MIB;
                total.pool_trimmed_mib += r.pool_trimmed / MIB;
            }
            Err(refusal) => return refusal,
        }
    }
    Json(total).into_response()
}

/// `GET /v1/models/status` — residency and footprint per registered slug.
pub async fn status(State(state): State<Arc<AppState>>) -> Response {
    let mut models = Vec::new();
    for slug in state.registry.slugs() {
        let residency = state.residency(&slug).as_str();
        let dp = state.dp_set(&slug);
        #[allow(unused_mut)]
        let mut entry = ModelStatus {
            resident: state.has_gpu_engine(&slug),
            serving: state.mux(&slug).is_some() && state.residency(&slug).admits()
                || dp.is_some_and(|set| set.ranks.iter().any(|r| r.is_up())),
            residency,
            required_mib: None,
            weights_mib: None,
            kv_mib: None,
            devices: None,
            group: None,
            model: slug.clone(),
            dp: dp.map(|set| {
                set.ranks
                    .iter()
                    .map(|r| {
                        let c = r.cand();
                        RankStatus {
                            rank: r.rank,
                            key: r.key.clone(),
                            device: r.ordinal,
                            group: r.group,
                            residency: state.residency(&r.key).as_str(),
                            resident: state.has_gpu_engine(&r.key),
                            serving: r.is_up(),
                            queued: c.pending.into(),
                            active: c.active.into(),
                            capacity: c.capacity.into(),
                            load: crate::serve::dp::load(&c),
                        }
                    })
                    .collect()
            }),
        };
        #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
        if let Some(set) = dp {
            let mgr = state.manager_for(&set.ranks[0].key);
            entry.required_mib = mgr
                .and_then(|m| m.required(&set.ranks[0].key))
                .map(|b| b / MIB);
            entry.devices = Some(set.ranks.iter().map(|r| r.ordinal).collect());
            models.push(entry);
            continue;
        }
        #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
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
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
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

    #[allow(unused_mut)]
    let mut capabilities = ControlCapabilities {
        load: false,
        unload: false,
        register: false,
        eviction: false,
    };
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    if !state.managers().is_empty() {
        capabilities = ControlCapabilities {
            load: true,
            unload: true,
            register: true,
            eviction: true,
        };
    }
    #[cfg(feature = "cpu")]
    if state.portable_manager().is_some() {
        capabilities = ControlCapabilities {
            load: true,
            unload: true,
            register: true,
            eviction: false,
        };
    }
    Json(StatusResponse {
        capabilities,
        models,
        groups,
    })
    .into_response()
}

/// One loadable bundle under a `--models-root`.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct AvailableModel {
    /// `<namespace>/<name>:<label>@g<n>` for a pin in a `$PLOW_HOME` store;
    /// `None` for a plain assets directory.
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    /// The slug `POST /v1/models/load` registers it under when none is
    /// given: the manifest's `served_name`, else its `network`.
    pub model: String,
    pub dir: PathBuf,
    pub bytes: u64,
    /// Registered in this server from this directory.
    pub loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct AvailableResponse {
    pub models: Vec<AvailableModel>,
}

/// `GET /v1/models/available` — the bundles under `--models-root` that
/// `load` would accept, so a caller can offer a pulled model without knowing
/// the node's disk layout. Scanned on every request: a `plowrt load` run
/// beside the server shows up without a restart.
pub async fn available(State(state): State<Arc<AppState>>) -> Response {
    let registered: Vec<PathBuf> = state
        .registry
        .slugs()
        .iter()
        .filter_map(|s| state.registry.get(s).ok())
        .filter_map(|b| b.dir.canonicalize().ok())
        .collect();
    let roots = state.models_roots().to_vec();
    match tokio::task::spawn_blocking(move || scan_models_roots(&roots, &registered)).await {
        Ok(models) => Json(AvailableResponse { models }).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// Every loadable bundle under `roots` (already canonical, as
/// `install_models_roots` leaves them).
///
/// A root that is a `$PLOW_HOME` store (or its `bundles/`) lists its pins,
/// with their refs; any other root lists itself and its immediate subdirectories that hold a
/// `weights.json` the registry can read. Each candidate is canonicalized and
/// must still sit under a root — the same test `resolve_assets` applies — so
/// a symlink cannot advertise a directory `load` would then refuse.
pub fn scan_models_roots(roots: &[PathBuf], registered: &[PathBuf]) -> Vec<AvailableModel> {
    let mut found: Vec<(Option<String>, PathBuf, Option<u64>)> = Vec::new();
    for root in roots {
        if let Some(store) = store_of(root) {
            if let Ok(store) = crate::dist::Store::open(store) {
                for p in store.list() {
                    found.push((Some(p.reference), p.dir, Some(p.bytes)));
                }
            }
            continue;
        }
        found.push((None, root.clone(), None));
        let Ok(rd) = std::fs::read_dir(root) else {
            continue;
        };
        let mut children: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        children.sort();
        found.extend(children.into_iter().map(|p| (None, p, None)));
    }

    let mut out: Vec<AvailableModel> = Vec::new();
    for (reference, dir, bytes) in found {
        let Ok(dir) = dir.canonicalize() else {
            continue;
        };
        if !roots.iter().any(|r| dir.starts_with(r)) || out.iter().any(|m| m.dir == dir) {
            continue;
        }
        let Some(model) = served_slug(&dir) else {
            continue;
        };
        out.push(AvailableModel {
            reference,
            model,
            bytes: bytes.unwrap_or_else(|| dir_bytes(&dir)),
            loaded: registered.contains(&dir),
            dir,
        });
    }
    out
}

/// The `$PLOW_HOME` store `root` is, or whose `bundles/` it is — the root
/// `serve --model` adds, as the parent of the bundle it started with.
fn store_of(root: &std::path::Path) -> Option<&std::path::Path> {
    let is_store = |p: &std::path::Path| p.join("refs").is_dir() && p.join("bundles").is_dir();
    if is_store(root) {
        return Some(root);
    }
    root.parent()
        .filter(|p| root.file_name() == Some("bundles".as_ref()) && is_store(p))
}

/// The slug `Registry::load` would give `dir`, or `None` when it has no
/// readable `weights.json` and so could not be registered at all.
fn served_slug(dir: &std::path::Path) -> Option<String> {
    let m: plow_asset::Manifest = crate::asset::read_json(&dir.join("weights.json")).ok()?;
    Some(m.served_name.unwrap_or(m.network))
}

/// Bytes of the regular files under `dir`. Symlinks are not followed: an
/// assets directory links its `checkpoint` in, and the weights are not the
/// bundle.
fn dir_bytes(dir: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("plow-available-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p.canonicalize().unwrap()
    }

    fn weights(network: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "network": network, "gpu": "Apple M4 Pro", "num_gpus": 1, "parallel": "tp",
            "weight_shared": false, "buckets": []
        }))
        .unwrap()
    }

    /// A pin in a `$PLOW_HOME` store: the bundle `plowrt load` leaves.
    fn pull_into(store_root: &std::path::Path, network: &str) -> PathBuf {
        use crate::dist::{Digest, Store};
        let st = Store::open(store_root).unwrap();
        let w = weights(network);
        let d = Digest::of(&w);
        st.put(&d, &w).unwrap();
        let dir = st
            .materialize("vid0", &[("weights.json".into(), d)])
            .unwrap();
        let bundle = serde_json::json!({
            "schema": plow_asset::dist::BUNDLE_SCHEMA, "namespace": "infervisor",
            "name": network, "label": "metal3-apple-m4-pro-tp1-2k", "generation": 2,
            "variant_id": "vid0", "plow_git": "0", "network": network,
            "target": {"vendor": "apple", "isa": "metal3", "sku": "Apple M4 Pro",
                       "units": 16, "mem_bytes": 1},
            "parallel": {"mode": "tp", "n": 1}, "max_ctx": 2048, "files": [],
            "tokenizer": {"source": "checkpoint", "file": "tokenizer.json"},
            "checkpoint": {"source": "hf:Qwen/Qwen3-0.6B", "revision": "r", "shards": 1,
                           "layout": "hf-safetensors", "bytes": 1}
        });
        st.record_bundle(&dir, &serde_json::from_value(bundle).unwrap())
            .unwrap();
        st.pin("file:///r/infervisor/qwen3-0.6b", "vid0").unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn a_plain_root_lists_its_loadable_subdirectories_and_nothing_outside() {
        let root = tmp("plain");
        let outside = tmp("outside");
        std::fs::create_dir_all(root.join("m1/hsaco")).unwrap();
        std::fs::write(root.join("m1/weights.json"), weights("m1")).unwrap();
        std::fs::write(root.join("m1/hsaco/a.elf"), [0u8; 10]).unwrap();
        // The farm link is the checkpoint, not the bundle: not counted.
        std::os::unix::fs::symlink(&outside, root.join("m1/checkpoint")).unwrap();
        std::fs::create_dir_all(root.join("notamodel")).unwrap();
        // A link out of the root is advertised nowhere: load would refuse it.
        std::fs::write(outside.join("weights.json"), weights("escaped")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();

        let m1 = root.join("m1");
        let got = scan_models_roots(&[root.clone()], &[m1.clone()]);
        let m1_bytes = weights("m1").len() as u64 + 10;
        assert_eq!(
            got,
            vec![AvailableModel {
                reference: None,
                model: "m1".into(),
                dir: m1,
                bytes: m1_bytes,
                loaded: true,
            }]
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn a_store_root_lists_its_pins_by_reference() {
        let root = tmp("store");
        let dir = pull_into(&root, "qwen3-0.6b");
        let want = vec![AvailableModel {
            reference: Some("infervisor/qwen3-0.6b:metal3-apple-m4-pro-tp1-2k@g2".into()),
            model: "qwen3-0.6b".into(),
            dir,
            bytes: weights("qwen3-0.6b").len() as u64,
            loaded: false,
        }];
        assert_eq!(scan_models_roots(&[root.clone()], &[]), want);
        // `serve --model` adds the store's `bundles/` as a root.
        assert_eq!(scan_models_roots(&[root.join("bundles")], &[]), want);
        assert!(scan_models_roots(&[], &[]).is_empty(), "no roots, nothing");
        let _ = std::fs::remove_dir_all(root);
    }
}
