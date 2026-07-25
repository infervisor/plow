//! §A Asset loading.
//!
//! A compiled model on disk is a directory containing `weights.json` (the
//! [`Manifest`]) plus, per shape bucket, a `.pkt` stream and its `.map.json` /
//! sidecars. [`ModelBundle::load`] reads the manifest and every bucket once at
//! startup; nothing here is on the hot path.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use plow_asset::Manifest;
pub use plow_asset::Phase;

use crate::text::tokenizer::{load_tokenizer, Tokenize};
use crate::{Result, RuntimeError};

mod bucket;
#[cfg(feature = "cuda")]
pub(crate) mod checkpoint;
pub mod devblob;
pub use bucket::{Bucket, BucketKey};

/// One compiled model: its manifest, every shape bucket (keyed by
/// `(phase, batch, seq)` for O(1) dispatch), and the model's tokenizer.
pub struct ModelBundle {
    /// Directory the assets were loaded from (weights resolved relative to it).
    pub dir: PathBuf,
    pub manifest: Manifest,
    buckets: HashMap<BucketKey, Bucket>,
    /// The model's tokenizer — a real HF `tokenizer.json` when present
    /// (feature `hf-tokenizer`), else the byte fallback.
    tokenizer: Arc<dyn Tokenize>,
}

impl ModelBundle {
    /// Load every artifact under `dir`. Validates each address map and packet
    /// stream up front so a bad asset fails loudly at startup, never mid-serve.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("weights.json");
        let manifest: Manifest = read_json(&manifest_path)?;

        let mut buckets = HashMap::with_capacity(manifest.buckets.len());
        for stat in &manifest.buckets {
            let key = BucketKey::new(&stat.phase, stat.batch, stat.seq);
            let bucket = Bucket::load(&dir, stat)?;
            buckets.insert(key, bucket);
        }

        // Load the model's tokenizer from `tokenizer.json` (byte fallback if
        // absent / feature off). Loaded once at startup, shared per request.
        let tokenizer = load_tokenizer(&dir);

        Ok(ModelBundle {
            dir,
            manifest,
            buckets,
            tokenizer,
        })
    }

    /// The model's advertised name (its API slug source).
    pub fn network(&self) -> &str {
        &self.manifest.network
    }

    /// The model's tokenizer (real HF tokenizer when available, else bytes).
    pub fn tokenizer(&self) -> &Arc<dyn Tokenize> {
        &self.tokenizer
    }

    /// Look up the compiled bucket serving `(phase, batch, seq)`.
    pub fn bucket(&self, key: BucketKey) -> Option<&Bucket> {
        self.buckets.get(&key)
    }

    /// Every compiled bucket key, for admission's bucket-ladder rounding (§I).
    pub fn bucket_keys(&self) -> impl Iterator<Item = BucketKey> + '_ {
        self.buckets.keys().copied()
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

/// Read + deserialize a JSON artifact, attributing errors to its path.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| RuntimeError::Json {
        path: path.to_path_buf(),
        source,
    })
}
