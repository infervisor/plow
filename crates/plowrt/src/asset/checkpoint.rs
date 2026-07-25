//! Safetensors checkpoint loading: mmap every `*.safetensors` shard once,
//! parse header metadata up front, and serve tensor bytes as zero-copy slices.

use std::ops::Range;
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::{Result, RuntimeError};

/// A safetensors checkpoint directory: every `*.safetensors` shard mmap'd,
/// with one metadata parse per shard (name → shard/offset resolved up front,
/// tensor bytes served as zero-copy mmap slices).
pub(crate) struct Checkpoint {
    shards: Vec<(memmap2::Mmap, usize)>, // (map, data-section offset)
    /// name → (shard index, byte range within the data section).
    index: FxHashMap<String, (usize, Range<usize>)>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        tracing::info!(dir = %dir.display(), "opening safetensors checkpoint...");
        let mut shards = Vec::new();
        let mut index = FxHashMap::default();
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|source| RuntimeError::Io { path: dir.to_path_buf(), source })?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file() && p.extension().map(|e| e == "safetensors").unwrap_or(false)
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(RuntimeError::Device(format!(
                "no *.safetensors in {}",
                dir.display()
            )));
        }
        tracing::info!(shards = paths.len(), "found safetensors shards");
        for path in &paths {
            let file = std::fs::File::open(path)
                .map_err(|source| RuntimeError::Io { path: path.clone(), source })?;
            // SAFETY: read-only checkpoint mapping held for the engine lifetime.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|source| RuntimeError::Io { path: path.clone(), source })?;
            let (header_len, meta) = safetensors::SafeTensors::read_metadata(&map)
                .map_err(|e| {
                    RuntimeError::Device(format!("safetensors {}: {e}", path.display()))
                })?;
            let data_off = 8 + header_len;
            let shard = shards.len();
            let tensors = meta.tensors();
            tracing::debug!(
                shard = shard,
                file = %path.display(),
                tensors = tensors.len(),
                mib = map.len() / (1 << 20),
                "mapped checkpoint shard"
            );
            index.reserve(tensors.len());
            for (name, info) in tensors {
                index.insert(name.clone(), (shard, info.data_offsets.0..info.data_offsets.1));
            }
            shards.push((map, data_off));
        }
        tracing::info!(
            shards = shards.len(),
            tensors = index.len(),
            "checkpoint ready (all shards mmap'd)"
        );
        Ok(Checkpoint { shards, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        let (shard, range) = self.index.get(name)?;
        let (map, off) = &self.shards[*shard];
        map.get(off + range.start..off + range.end)
    }
}
