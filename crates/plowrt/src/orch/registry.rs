//! §F Model registry — slug → loaded bundle. Many models share the device pools
//! and persistent kernels (weights differ, kernels don't).
//!
//! Interior mutability is load-bearing, not a style choice: the control plane
//! registers and drops bundles while the server is live (`POST /v1/models/load`),
//! and `AppState` is behind an `Arc` shared by every in-flight request, so a
//! `&mut self` API could only ever be driven at startup. `get` hands back a
//! cloned `Arc`, so no reader holds the lock across a request.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::asset::ModelBundle;
use crate::{Result, RuntimeError};

/// The set of loaded models, keyed by API slug.
#[derive(Default)]
pub struct Registry {
    models: RwLock<FxHashMap<String, Arc<ModelBundle>>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry::default()
    }

    /// Load a bundle from `dir` and register it under `slug` (defaults to the
    /// manifest network name). Returns the slug it was registered under.
    ///
    /// A slug already in the registry is an ERROR, not an overwrite. Two
    /// bundles whose manifests carry the same network name used to silently
    /// drop the first one — the server then listed one model and served the
    /// other's weights under it, with nothing in the log.
    pub fn load(&self, dir: impl AsRef<Path>, slug: Option<String>) -> Result<String> {
        let bundle = ModelBundle::load(dir)?;
        let slug = slug.unwrap_or_else(|| bundle.network().to_string());
        let mut models = self.models.write();
        if models.contains_key(&slug) {
            return Err(RuntimeError::Msg(format!(
                "model slug {slug:?} is already registered — pass a distinct \
                 slug (two bundles with the same network name need one each)"
            )));
        }
        models.insert(slug.clone(), Arc::new(bundle));
        Ok(slug)
    }

    /// Resolve a request `model` slug to its bundle.
    pub fn get(&self, slug: &str) -> Result<Arc<ModelBundle>> {
        self.models
            .read()
            .get(slug)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownModel(slug.to_string()))
    }

    /// Whether `slug` is registered.
    pub fn contains(&self, slug: &str) -> bool {
        self.models.read().contains_key(slug)
    }

    /// Unload a model by slug. The `Arc<ModelBundle>` is removed from the
    /// registry; when all remaining references (mux tasks, in-flight requests)
    /// drop, the bundle's device memory is released. Returns the bundle for
    /// the caller to orchestrate drain if needed.
    pub fn unload(&self, slug: &str) -> Result<Arc<ModelBundle>> {
        self.models
            .write()
            .remove(slug)
            .ok_or_else(|| RuntimeError::UnknownModel(slug.to_string()))
    }

    /// Registered slugs — backs `GET /v1/models`. Sorted, so the listing is
    /// stable across scrapes instead of following hash order.
    pub fn slugs(&self) -> Vec<String> {
        let mut v: Vec<String> = self.models.read().keys().cloned().collect();
        v.sort_unstable();
        v
    }

    pub fn len(&self) -> usize {
        self.models.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.read().is_empty()
    }
}
