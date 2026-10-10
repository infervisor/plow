//! Explicit residency control for native CPU and Metal slot engines.
use super::{
    admin::{err, LoadRequest, LoadResponse, UnloadRequest},
    engine::CpuServe,
    mux::{self, MuxConfig},
    AppState, Residency,
};
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

pub struct PortableManager {
    mux: MuxConfig,
    checkpoints: RwLock<FxHashMap<String, PathBuf>>,
}
impl PortableManager {
    pub fn new(mux: MuxConfig) -> Self {
        Self {
            mux,
            checkpoints: RwLock::new(FxHashMap::default()),
        }
    }
    pub fn register(&self, slug: String, checkpoint: PathBuf) {
        self.checkpoints.write().insert(slug, checkpoint);
    }

    pub async fn load(
        &self,
        state: &Arc<AppState>,
        req: LoadRequest,
        assets: Option<PathBuf>,
    ) -> Response {
        if req.evict || req.device.is_some_and(|d| d != 0) {
            return err(
                StatusCode::BAD_REQUEST,
                "CPU/Metal control supports device 0 without automatic eviction",
            );
        }
        let known = state.registry.contains(&req.model);
        if known && !self.checkpoints.read().contains_key(&req.model) {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                "model is not managed by the CPU/Metal slot engine",
            );
        }
        let dir = match (assets, state.registry.get(&req.model)) {
            (Some(dir), Ok(bundle)) if bundle.dir.canonicalize().ok().as_ref() != Some(&dir) => {
                return err(
                    StatusCode::CONFLICT,
                    "model has different assets; deregister first",
                )
            }
            (Some(dir), _) => dir,
            (None, Ok(bundle)) => bundle.dir.clone(),
            _ => {
                return err(
                    StatusCode::NOT_FOUND,
                    "unknown model; supply assets to register it",
                )
            }
        };
        let saved = self.checkpoints.read().get(&req.model).cloned();
        let checkpoint = req
            .checkpoint
            .as_ref()
            .map(PathBuf::from)
            .or(saved.clone())
            .unwrap_or_else(|| crate::asset::serve::checkpoint_dir(&dir));
        if saved.as_ref().is_some_and(|p| p != &checkpoint) {
            return err(
                StatusCode::CONFLICT,
                "model has a different checkpoint; deregister first",
            );
        }
        for alias in &req.aliases {
            if state.registry.contains(alias)
                || state
                    .registry
                    .resolve(alias)
                    .is_some_and(|s| s != req.model)
            {
                return err(StatusCode::CONFLICT, "alias is already assigned");
            }
        }
        let started = Instant::now();
        if !state.has_gpu_engine(&req.model) {
            let st = state.clone();
            let slug = req.model.clone();
            let ckpt = checkpoint.clone();
            let loaded = tokio::task::spawn_blocking(move || -> Result<_, String> {
                if !known {
                    st.registry
                        .load(&dir, Some(slug.clone()))
                        .map_err(|e| e.to_string())?;
                }
                let result = (|| {
                    let bundle = st.registry.get(&slug).map_err(|e| e.to_string())?;
                    if bundle.tokenizer().is_byte_fallback() {
                        return Err("a real tokenizer is required".into());
                    }
                    let blob = crate::asset::devblob::DevBlob::find_in_dir(&dir)
                        .map_err(|e| e.to_string())?
                        .ok_or("no compiled model packet")?;
                    load_engine(&blob, &ckpt).map_err(|e| e.to_string())
                })();
                if result.is_err() && !known {
                    let _ = st.registry.unload(&slug);
                }
                result
            })
            .await;
            let engine = match loaded {
                Ok(Ok(e)) => e,
                Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, e),
                Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            };
            self.register(req.model.clone(), checkpoint);
            state.install_gpu_engine(req.model.clone(), super::engine::ServeEngine::Cpu(engine));
        }
        if state.mux(&req.model).is_none() {
            let bundle = match state.registry.get(&req.model) {
                Ok(b) => b,
                Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            };
            let m = mux::spawn(req.model.clone(), bundle, state.clone(), self.mux);
            state.install_mux(req.model.clone(), m);
        }
        for alias in req.aliases {
            if let Err(e) = state.registry.add_alias(alias, &req.model) {
                return err(StatusCode::CONFLICT, e);
            }
        }
        state.set_residency(&req.model, Residency::Auto);
        Json(LoadResponse {
            model: req.model,
            state: "resident",
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
        .into_response()
    }

    pub async fn unload(&self, state: &Arc<AppState>, req: UnloadRequest) -> Response {
        if !self.checkpoints.read().contains_key(&req.model) {
            return err(
                StatusCode::NOT_FOUND,
                "model is not managed by the CPU/Metal slot engine",
            );
        }
        state.set_residency(&req.model, Residency::Unloading);
        let started = Instant::now();
        if let Some(m) = state.remove_mux(&req.model) {
            m.preempt().await;
        }
        let stop_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let engine = state.remove_gpu_engine(&req.model);
        if let Err(e) = tokio::task::spawn_blocking(move || drop(engine)).await {
            return err(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
        state.set_residency(&req.model, Residency::Unloaded);
        if req.deregister {
            let _ = state.registry.unload(&req.model);
            self.checkpoints.write().remove(&req.model);
            state.clear_slug_group(&req.model);
            state.set_residency(&req.model, Residency::Auto);
        }
        // Native backends do not yet expose reliable allocation accounting.
        Json(serde_json::json!({"model":req.model,"state":"unloaded","stop_ms":stop_ms,
            "unload_ms":started.elapsed().as_secs_f64()*1000.0,"freed_mib":null,"pool_trimmed_mib":null,"deregistered":req.deregister})).into_response()
    }
}

pub fn load_engine(blob: &Path, checkpoint: &Path) -> crate::Result<CpuServe> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if crate::config::RuntimeConfig::get().apple.backend.as_deref() != Some("cpu") {
        let engine = crate::exec::apple::MetalEngine::load(blob, checkpoint)?;
        return CpuServe::from_engine(Box::new(engine), checkpoint);
    }
    let cpu = &crate::config::RuntimeConfig::get().cpu;
    let opts = crate::exec::cpu::engine::CpuEngineOpts {
        threads: cpu.threads as usize,
        numa: cpu.numa.clone(),
        spin_us: cpu.spin_us,
        topology: None,
        isa: match cpu.isa {
            crate::config::CpuIsa::Scalar => crate::exec::cpu::ffi::Isa::Scalar,
            crate::config::CpuIsa::Avx512 => crate::exec::cpu::ffi::Isa::Avx512,
            _ => crate::exec::cpu::ffi::Isa::Amx,
        },
    };
    CpuServe::load(blob, checkpoint, &opts)
}
