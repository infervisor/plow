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
    app(state)
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
