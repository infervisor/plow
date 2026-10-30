//! §I.5 control plane — residency status, and the guards on `load`/`unload`.
//!
//! CPU backend, so no engine is ever resident here. What these cover is the
//! part that must hold on any server: the allow-list defaults closed, an
//! unmanaged slug is never answered with a success, and status reports the
//! residency of every registered model rather than only the loaded ones.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, MuxConfig};
use plowrt::serve::{app, AppState};
use tower::ServiceExt;

mod common;

use std::sync::atomic::{AtomicU32, Ordering};
static DIR_SEQ: AtomicU32 = AtomicU32::new(0);

/// The CPU-backed AppState the router is built over, for assertions that are
/// about state wiring rather than HTTP.
fn make_state() -> Arc<AppState> {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_admin_state_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle(&dir, "state-model");

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    Arc::new(AppState::new(registry, execset))
}

/// Co-tenant scheduling is host-side sequencing of ticks, not a vendor
/// feature. It used to be compiled out unless the CUDA feature was on, which
/// silently disabled it on AMD — where two models already share one ROCr agent
/// and HSA has no cooperative-launch refusal to catch the oversubscription —
/// and on CPU, where each model owns its own worker pool.
#[test]
fn device_turns_are_available_without_a_gpu_backend() {
    let state = make_state();
    // Nothing installed yet: no turn to take.
    assert!(state.device_turn("state-model").is_none());

    state.install_device_turns(1, false);
    assert!(
        state.device_turn("state-model").is_some(),
        "a CPU serve got no co-tenant turn — the mechanism was compiled out"
    );
    // A slug with no recorded group falls back to group 0 rather than to
    // "unordered", which is what the AMD and CPU paths rely on.
    assert!(state.device_turn("never-registered").is_some());
}

/// Group mapping is not a CUDA concept either — every backend serves a device
/// set, they just mostly have one.
#[test]
fn slug_group_mapping_works_without_a_gpu_backend() {
    let state = make_state();
    state.install_device_turns(2, false);
    assert_eq!(state.slug_group("state-model"), None);
    state.set_slug_group("state-model", 1);
    assert_eq!(state.slug_group("state-model"), Some(1));
    assert!(state.device_turn("state-model").is_some());
    state.clear_slug_group("state-model");
    assert_eq!(state.slug_group("state-model"), None);
}

fn make_app() -> axum::Router {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_admin_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle(&dir, "admin-model");

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    for slug in state.registry.slugs() {
        let bundle = state.registry.get(&slug).unwrap();
        let m = mux::spawn(
            slug.clone(),
            bundle,
            Arc::clone(&state),
            MuxConfig::default(),
        );
        state.install_mux(slug, m);
    }
    app(Arc::clone(&state)).merge(plowrt::serve::admin_app(state))
}

async fn body_string(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn post(uri: &str, body: &str) -> axum::response::Response {
    make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn status_reports_residency_for_every_registered_model() {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    let m = v["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|m| m["model"] == "admin-model")
        .expect("admin-model listed");
    // Nothing was unloaded, so the slug is unpinned; the CPU reference path
    // installs a dispatcher but never a device engine.
    assert_eq!(m["residency"], "auto");
    assert_eq!(m["resident"], false);
    assert_eq!(m["serving"], true);
}

/// The allow-list defaults CLOSED. A server started with no `--models-root`
/// must refuse an assets dir named in a request body: loading a bundle means
/// executing the cubins in that directory.
#[tokio::test]
async fn load_refuses_an_assets_dir_when_no_models_root_is_configured() {
    let resp = post("/v1/models/load", r#"{"model":"evil","assets":"/etc"}"#).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert!(body_string(resp).await.contains("models-root"));
}

/// `available` lists what `load` would accept under the roots, and marks the
/// one this server already registered from its directory.
#[tokio::test]
async fn available_lists_bundles_under_the_models_roots() {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("plowrt_available_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    common::write_bundle(&root.join("served"), "served-model");
    common::write_bundle(&root.join("pulled"), "pulled-model");

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(root.join("served"), None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    state.install_models_roots([root.clone()]);

    let resp = plowrt::serve::admin_app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models/available")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    let models = v["models"].as_array().expect("models array");
    let by = |slug: &str| models.iter().find(|m| m["model"] == slug).cloned();
    let pulled = by("pulled-model").expect("pulled-model listed");
    assert_eq!(
        pulled["ref"],
        serde_json::Value::Null,
        "a plain dir is no pin"
    );
    assert_eq!(pulled["loaded"], false);
    assert!(pulled["bytes"].as_u64().unwrap() > 0);
    let canon = root.join("pulled").canonicalize().unwrap();
    assert_eq!(pulled["dir"], canon.to_str().unwrap());
    assert_eq!(
        by("served-model").expect("served-model listed")["loaded"],
        true
    );
    let _ = std::fs::remove_dir_all(root);
}

/// `GET /v1/models` is the OpenAI catalogue and must keep listing a slug
/// whatever its residency — clients must not see models appear and disappear
/// because of a placement decision.
#[tokio::test]
async fn the_openai_model_list_is_not_a_residency_report() {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("admin-model"));
    // No residency vocabulary leaks into the OpenAI surface.
    assert!(!body.contains("residency"));
    assert!(!body.contains("resident"));
}

/// A control-plane call that silently does nothing is worse than an error, so
/// unload on a server with no model manager must not answer 200.
#[tokio::test]
async fn unload_without_a_manager_is_not_a_success() {
    let resp = post("/v1/models/unload", r#"{"model":"admin-model"}"#).await;
    assert_ne!(resp.status(), StatusCode::OK);
}

/// Likewise load: an unknown slug with no assets dir cannot quietly succeed.
#[tokio::test]
async fn load_of_an_unknown_slug_is_not_a_success() {
    let resp = post("/v1/models/load", r#"{"model":"no-such-model"}"#).await;
    assert_ne!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn public_router_does_not_expose_model_control() {
    let state = make_state();
    for (method, uri) in [
        ("POST", "/v1/models/load"),
        ("POST", "/v1/models/unload"),
        ("GET", "/v1/models/status"),
        ("GET", "/v1/models/available"),
    ] {
        let response = app(state.clone())
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"state-model"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{uri} is public");
    }
}

#[tokio::test]
async fn lifecycle_operations_serialize_per_model_without_blocking_other_models() {
    let state = make_state();
    let first = state.control_lock("a").await;
    let other = tokio::time::timeout(std::time::Duration::from_secs(1), state.control_lock("b"))
        .await
        .unwrap();
    let s = state.clone();
    let mut same = tokio::spawn(async move { s.control_lock("a").await });
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut same)
            .await
            .is_err()
    );
    drop(first);
    drop(
        tokio::time::timeout(std::time::Duration::from_secs(1), same)
            .await
            .unwrap()
            .unwrap(),
    );
    drop(other);
}

#[tokio::test]
async fn version_endpoint_answers_with_current_version() {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .uri("/version")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
}

#[test]
fn unload_request_deserializes_drain_timeout() {
    let req: plowrt::serve::admin::UnloadRequest = serde_json::from_str(
        r#"{"model":"target-model","deregister":true,"drain_timeout_ms":15000}"#,
    )
    .unwrap();
    assert_eq!(req.model, "target-model");
    assert!(req.deregister);
    assert_eq!(req.drain_timeout_ms, Some(15000));
}

#[tokio::test]
async fn endpoints_are_cached_on_bundle_at_registration() {
    let state = make_state();
    let bundle = state.registry.get("state-model").unwrap();
    assert_eq!(bundle.endpoints(), &["chat/completions", "completions"]);
    assert!(!bundle.is_speech());
    assert!(!bundle.is_audio());
}

#[cfg(feature = "cpu")]
#[tokio::test]
async fn portable_manager_unload_honors_drain_timeout() {
    let state = make_state();
    let mgr = plowrt::serve::portable::PortableManager::new(MuxConfig::default());
    mgr.register(
        "state-model".into(),
        std::path::PathBuf::from("/nonexistent"),
    );
    state.install_portable_manager(mgr);

    let req = plowrt::serve::admin::UnloadRequest {
        model: "state-model".into(),
        deregister: false,
        rank: None,
        device: None,
        drain_timeout_ms: Some(10),
    };
    let resp = state.portable_manager().unwrap().unload(&state, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn unloaded_asr_model_reports_audio_endpoints_in_catalogue() {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_asr_test_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle(&dir, "asr-model");
    std::fs::write(dir.join("audio_lm_contract.json"), b"{}").unwrap();

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    let app = app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    let val: serde_json::Value = serde_json::from_str(&body).unwrap();
    let m = val["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "asr-model")
        .expect("asr-model in catalogue");
    assert_eq!(
        m["x_plow_endpoints"],
        serde_json::json!(["audio/transcriptions", "audio/transcriptions/stream"])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn server_brings_up_cleanly_with_zero_assets() {
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    let state = Arc::new(AppState::new(registry, execset));
    let router = app(Arc::clone(&state)).merge(plowrt::serve::admin_app(Arc::clone(&state)));

    let resp = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = body_string(resp).await;
    assert_eq!(status, StatusCode::OK, "response body: {body}");
    let val: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(val["data"].as_array().unwrap().len(), 0);

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/v1/models/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn load_without_assets_field_finds_local_bundle_under_models_root() {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("plowrt_root_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    common::write_bundle(&root.join("dynamic-model"), "dynamic-model");

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    let state = Arc::new(AppState::new(registry, execset));
    state.install_models_roots(vec![root.clone()]);

    let router = plowrt::serve::admin_app(Arc::clone(&state));

    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"dynamic-model"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = resp.status();
    let body = body_string(resp).await;
    // Without a GPU manager, the request successfully resolved the local bundle under models_root
    // (advancing past 404 NOT_FOUND) and reached the manager stage (501 NOT_IMPLEMENTED).
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "response body: {body}");
    assert!(body.contains("no model manager on this server"));

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn load_without_assets_field_and_no_models_root_is_refused() {
    let resp = post("/v1/models/load", r#"{"model":"non-existent-anywhere"}"#).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let body = body_string(resp).await;
    assert!(body.contains("no --models-root"), "{body}");
}

/// A root holding one loadable bundle, `inside`, plus a bundle `outside` it, and an admin
/// router whose only allow-list entry is the root.
fn rooted(tag: &str) -> (std::path::PathBuf, std::path::PathBuf, axum::Router) {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("plowrt_roots_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let root = base.join("root");
    let outside = base.join("outside");
    common::write_bundle(&root.join("inside"), "inside");
    common::write_bundle(&outside, "outside");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(2));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let state = Arc::new(AppState::new(Registry::new(), execset));
    state.install_models_roots(vec![root.clone()]);
    (root, outside, plowrt::serve::admin_app(state))
}

async fn load_model(router: axum::Router, model: &str) -> (StatusCode, String) {
    let body = serde_json::json!({ "model": model }).to_string();
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    (status, body_string(resp).await)
}

/// `model` without `assets` names a bundle UNDER a root. An absolute path, a `..` walk and a
/// symlink out of the root used to be loaded from wherever they pointed — executing its cubins
/// with the allow-list never consulted. Reaching the manager stage (501 here, no GPU) is the
/// tell that the path was accepted.
#[tokio::test]
async fn load_by_name_never_leaves_the_models_roots() {
    let (root, outside, router) = rooted("escape");
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    let absolute = outside.display().to_string();
    for model in [
        absolute.as_str(),
        "../outside",
        "inside/../../outside",
        "link",
    ] {
        let (status, body) = load_model(router.clone(), model).await;
        assert!(
            status == StatusCode::FORBIDDEN || status == StatusCode::NOT_FOUND,
            "{model}: {status} {body}"
        );
        assert!(
            !body.contains("no model manager"),
            "{model} was accepted: {body}"
        );
    }
    let (status, body) = load_model(router, "inside").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

/// An explicit `assets` dir is held to the same test, symlinks included.
#[tokio::test]
async fn load_with_assets_through_a_symlink_out_of_the_root_is_refused() {
    let (root, outside, router) = rooted("assets-link");
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    let body = serde_json::json!({
        "model": "m", "assets": root.join("link").display().to_string()
    })
    .to_string();
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/load")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

/// The startup allow-list: `--models-root`, the parents of the `--assets` dirs (a single-asset
/// serve reaches its siblings with no flag), the store's `bundles/` — and never the working
/// directory, whatever it is.
#[test]
fn startup_models_roots_are_explicit_roots_asset_parents_and_the_store() {
    use plowrt::serve::admin::startup_models_roots;
    use std::path::PathBuf;
    let cwd = std::env::current_dir().unwrap();
    let single = startup_models_roots(&[], &[PathBuf::from("/srv/models/e4b")], None);
    assert_eq!(single, vec![PathBuf::from("/srv/models")]);
    let all = startup_models_roots(
        &["/data/roots".into()],
        &[
            PathBuf::from("/srv/models/e4b"),
            PathBuf::from("/srv/models/g12"),
        ],
        Some(PathBuf::from("/home/u/.plow/bundles")),
    );
    assert_eq!(
        all,
        vec![
            PathBuf::from("/data/roots"),
            PathBuf::from("/srv/models"),
            PathBuf::from("/home/u/.plow/bundles"),
        ]
    );
    assert!(startup_models_roots(&[], &[], None).is_empty());
    assert!(!all.contains(&cwd) && !all.contains(&cwd.join("models")));
}
