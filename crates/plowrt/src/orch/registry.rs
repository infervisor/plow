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

fn vendor_of(b: &ModelBundle) -> Option<hwspec::Vendor> {
    hwspec::registry::lookup(&b.manifest.gpu).map(|s| s.vendor)
}

/// The set of loaded models, keyed by API slug.
#[derive(Default)]
pub struct Registry {
    models: RwLock<FxHashMap<String, Arc<ModelBundle>>>,
    /// Extra names a registered model answers to (vLLM's `--served-model-name`),
    /// alias -> canonical slug. Clients routinely hardcode a model name they
    /// cannot change; without this the only way to serve them was to rename the
    /// bundle, which changes every metric label and dashboard along with it.
    aliases: RwLock<FxHashMap<String, String>>,
    /// Other vendors' bundles of a registered model (docs/arch/30 §3.1): slug -> bundles whose
    /// packets target another vendor than the registered one, checked to answer identically.
    variants: RwLock<FxHashMap<String, Vec<Arc<ModelBundle>>>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry::default()
    }

    /// Load a bundle from `dir` and register it under `slug` (defaults to
    /// `Manifest::default_slug`). Returns the slug it was registered under.
    ///
    /// A slug already in the registry is an ERROR, not an overwrite. Two
    /// bundles whose manifests carry the same network name used to silently
    /// drop the first one — the server then listed one model and served the
    /// other's weights under it, with nothing in the log.
    pub fn load(&self, dir: impl AsRef<Path>, slug: Option<String>) -> Result<String> {
        let bundle = ModelBundle::load(dir)?;
        let slug = slug.unwrap_or_else(|| bundle.manifest.default_slug(&bundle.dir));
        let mut models = self.models.write();
        if let Some(first) = models.get(&slug) {
            // A bundle for ANOTHER vendor under a registered name is that model's variant, when
            // it answers identically; anything else is the duplicate refused below.
            let (va, vb) = (vendor_of(first), vendor_of(&bundle));
            if va.is_some() && vb.is_some() && va != vb {
                if let Some(why) = crate::serve::chat::variant_mismatch(first, &bundle) {
                    return Err(RuntimeError::Msg(format!(
                        "{}: cannot serve as a {vb:?} variant of {slug:?} ({}): {why}",
                        bundle.dir.display(),
                        first.dir.display()
                    )));
                }
                let mut variants = self.variants.write();
                let list = variants.entry(slug.clone()).or_default();
                if list.iter().any(|b| vendor_of(b) == vb) {
                    return Err(RuntimeError::Msg(format!(
                        "model {slug:?} already has a {vb:?} variant"
                    )));
                }
                list.push(Arc::new(bundle));
                return Ok(slug);
            }
        }
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

    /// `slug`'s bundle for `vendor`: the registered one when it targets `vendor`, else a variant.
    pub fn variant(&self, slug: &str, vendor: hwspec::Vendor) -> Option<Arc<ModelBundle>> {
        let primary = self.models.read().get(slug).cloned()?;
        if vendor_of(&primary) == Some(vendor) {
            return Some(primary);
        }
        self.variants
            .read()
            .get(slug)?
            .iter()
            .find(|b| vendor_of(b) == Some(vendor))
            .cloned()
    }

    /// `slug`'s bundle for a device of `vendor`: its variant there when it has one, else the
    /// registered bundle.
    pub fn bundle_for(
        &self,
        slug: &str,
        vendor: Option<hwspec::Vendor>,
    ) -> Result<Arc<ModelBundle>> {
        match vendor.and_then(|v| self.variant(slug, v)) {
            Some(b) => Ok(b),
            None => self.get(slug),
        }
    }

    /// The vendors `slug` has a bundle for, the registered one first.
    pub fn vendors(&self, slug: &str) -> Vec<hwspec::Vendor> {
        let Some(primary) = self.models.read().get(slug).cloned() else {
            return Vec::new();
        };
        let mut out: Vec<_> = vendor_of(&primary).into_iter().collect();
        if let Some(list) = self.variants.read().get(slug) {
            out.extend(list.iter().filter_map(|b| vendor_of(b)));
        }
        out
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
        self.aliases
            .write()
            .retain(|_, canonical| canonical != slug);
        self.variants.write().remove(slug);
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
