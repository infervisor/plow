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
    /// name → where the bytes are, and what shape they are.
    index: FxHashMap<String, Entry>,
}

/// One tensor's location and shape.
///
/// The shape is kept because a ROW-parallel shard cannot be expressed without
/// it. A column shard is a contiguous byte range and needs only sizes; a row
/// shard of an `[out, in]` matrix is a *strided column range* — `in/N`
/// contiguous elements out of every one of `out` rows — and the stride IS `in`.
/// Nothing else in the byte count recovers it: `[4096, 5376]` and `[5376, 4096]`
/// have identical lengths and gather to different tensors.
struct Entry {
    shard: usize,
    range: Range<usize>,
    shape: Vec<usize>,
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
                index.insert(
                    name.clone(),
                    Entry {
                        shard,
                        range: info.data_offsets.0..info.data_offsets.1,
                        shape: info.shape.clone(),
                    },
                );
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

    /// Tensor bytes alone. The CUDA engine binds full tensors and wants only
    /// these; the AMD engine binds shards and needs [`Checkpoint::tensor_ex`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensor_ex(name).map(|(bytes, _)| bytes)
    }

    /// Tensor bytes **and** shape — what a row-parallel shard needs.
    pub fn tensor_ex(&self, name: &str) -> Option<(&[u8], &[usize])> {
        let e = self.index.get(name)?;
        let (map, off) = &self.shards[e.shard];
        let bytes = map.get(off + e.range.start..off + e.range.end)?;
        Some((bytes, &e.shape))
    }
}

/// The checkpoint's stop-token set: `generation_config.json` `eos_token_id`
/// (int or list), falling back to `config.json`, falling back to empty (the
/// caller then stops on max_tokens only).
///
/// Metadata, not tensors — so it reads the directory directly and does not need
/// the shards mmap'd. Shared by both engines: a backend that skipped this emits
/// its eos id as ordinary text and runs every request to `max_tokens`.
pub(crate) fn read_eos_ids(dir: &Path) -> Vec<u32> {
    for file in ["generation_config.json", "config.json"] {
        let Ok(bytes) = std::fs::read(dir.join(file)) else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    return vec![id as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> =
                    a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}
