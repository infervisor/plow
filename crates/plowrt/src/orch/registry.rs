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
    /// Extra names a registered model answers to (vLLM's `--served-model-name`),
    /// alias -> canonical slug. Clients routinely hardcode a model name they
    /// cannot change; without this the only way to serve them was to rename the
    /// bundle, which changes every metric label and dashboard along with it.
    aliases: RwLock<FxHashMap<String, String>>,
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

    /// Register `alias` as another name for `canonical`.
    ///
    /// Refuses an alias that collides with a registered slug or another alias:
    /// silently shadowing one model with another is the failure `load` already
    /// refuses for duplicate slugs, reached by a different route.
    pub fn add_alias(&self, alias: String, canonical: &str) -> Result<()> {
        if !self.contains(canonical) {
            return Err(RuntimeError::UnknownModel(canonical.to_string()));
        }
        if self.contains(&alias) {
            return Err(RuntimeError::Msg(format!(
                "alias {alias:?} is already a registered model slug"
            )));
        }
        let mut aliases = self.aliases.write();
        if let Some(existing) = aliases.get(&alias) {
            if existing != canonical {
                return Err(RuntimeError::Msg(format!(
                    "alias {alias:?} already points at {existing:?}"
                )));
            }
            return Ok(());
        }
        aliases.insert(alias, canonical.to_string());
        Ok(())
    }

    /// The canonical slug `name` refers to, when it is an ALIAS. `None` when
    /// `name` is already a slug (or is unknown) — the caller then uses it
    /// unchanged, so the common path costs one lookup and no allocation.
    pub fn resolve(&self, name: &str) -> Option<String> {
        self.aliases.read().get(name).cloned()
    }

    /// Every (alias, canonical) pair, sorted by alias.
    pub fn alias_pairs(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .aliases
            .read()
            .iter()
            .map(|(a, c)| (a.clone(), c.clone()))
            .collect();
        v.sort_unstable();
        v
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
        let bundle = self
            .models
            .write()
            .remove(slug)
            .ok_or_else(|| RuntimeError::UnknownModel(slug.to_string()))?;
        // Otherwise the alias outlives its target and resolves to a slug that
        // is no longer registered — a 404 naming a model the client never asked
        // for.
        self.aliases.write().retain(|_, canonical| canonical != slug);
        Ok(bundle)
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
