//! §F Model registry — slug → loaded bundle. Many models share the device pools
//! and persistent kernels (weights differ, kernels don't).

use std::path::Path;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::asset::ModelBundle;
use crate::{Result, RuntimeError};

/// The set of loaded models, keyed by API slug.
#[derive(Default)]
pub struct Registry {
    models: FxHashMap<String, Arc<ModelBundle>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry::default()
    }

    /// Load a bundle from `dir` and register it under `slug` (defaults to the
    /// manifest network name). Returns the slug it was registered under.
    pub fn load(&mut self, dir: impl AsRef<Path>, slug: Option<String>) -> Result<String> {
        let bundle = ModelBundle::load(dir)?;
        let slug = slug.unwrap_or_else(|| bundle.network().to_string());
        self.models.insert(slug.clone(), Arc::new(bundle));
        Ok(slug)
    }

    /// Resolve a request `model` slug to its bundle.
    pub fn get(&self, slug: &str) -> Result<Arc<ModelBundle>> {
        self.models
            .get(slug)
            .cloned()
            .ok_or_else(|| RuntimeError::UnknownModel(slug.to_string()))
    }

    /// Unload a model by slug. The `Arc<ModelBundle>` is removed from the
    /// registry; when all remaining references (mux tasks, in-flight requests)
    /// drop, the bundle's device memory is released. Returns the bundle for
    /// the caller to orchestrate drain if needed.
    pub fn unload(&mut self, slug: &str) -> Result<Arc<ModelBundle>> {
        self.models
            .remove(slug)
            .ok_or_else(|| RuntimeError::UnknownModel(slug.to_string()))
    }

    /// Registered slugs — backs `GET /v1/models`.
    pub fn slugs(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}
