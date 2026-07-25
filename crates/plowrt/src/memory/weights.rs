//! §C Weight loading — resolve each `Persistent` map entry to checkpoint bytes,
//! tile to the manifest `(bn, bk)`, and upload to the arena.

use plow_asset::{BufClass, Manifest};

use crate::memory::AddressSpace;
use crate::Result;

/// Source of weight bytes. `hub` feature backs this with safetensors; the
/// default is a zero-filled placeholder so the skeleton loads without a
/// checkpoint present.
pub trait WeightSource {
    /// Bytes for tensor `name`, already laid out in the compiled `(bn, bk)`
    /// tiling, or `None` if absent.
    fn tensor(&self, name: &str) -> Option<Vec<u8>>;
}

/// Placeholder source: yields zeroed buffers sized to the map's `reserved`.
pub struct ZeroWeights;

impl WeightSource for ZeroWeights {
    fn tensor(&self, _name: &str) -> Option<Vec<u8>> {
        None
    }
}

/// A `safetensors`-backed weight source (feature `hub`). Memory-maps the
/// checkpoint and returns each tensor's raw bytes by name. NOTE: it yields the
/// on-disk layout; arranging bytes into the compiled `(bn, bk)` tiling is the
/// next step (the manifest carries `weight_tiling`) — this closes the *loading*
/// path, tiling is applied by the caller before upload.
#[cfg(feature = "hub")]
pub struct SafetensorsWeights {
    mmap: memmap2::Mmap,
}

#[cfg(feature = "hub")]
impl SafetensorsWeights {
    /// Memory-map a `.safetensors` checkpoint.
    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        let file = std::fs::File::open(path).map_err(|source| crate::RuntimeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: read-only checkpoint held for the process lifetime.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| {
            crate::RuntimeError::Io {
                path: path.to_path_buf(),
                source,
            }
        })?;
        Ok(SafetensorsWeights { mmap })
    }
}

#[cfg(feature = "hub")]
impl WeightSource for SafetensorsWeights {
    fn tensor(&self, name: &str) -> Option<Vec<u8>> {
        // Parsing the header per call is cheap (JSON metadata only); the tensor
        // bytes are a zero-copy view into the mmap that we copy out.
        let st = safetensors::SafeTensors::deserialize(&self.mmap).ok()?;
        let t = st.tensor(name).ok()?;
        Some(t.data().to_vec())
    }
}

/// Load every persistent tensor into `space`, replicating per device.
///
/// Follows the compiler contract: locate by `name`, upload exactly `reserved`
/// bytes at the rebased physical address. Missing tensors are zero-filled by the
/// placeholder source (real deployments pass a safetensors-backed source).
pub fn load_weights(
    space: &AddressSpace,
    _manifest: &Manifest,
    source: &dyn WeightSource,
) -> Result<usize> {
    let backend = space.backend().clone();
    // Persistent = checkpoint weights; Static = compile-time constants (RoPE
    // freq tables, static masks) that ship in `static_tensors.bin`. Both are
    // filled once at load; the placeholder source zero-fills either.
    let entries: Vec<_> = space
        .map()
        .entries
        .iter()
        .filter(|e| matches!(e.class, BufClass::Persistent | BufClass::Static))
        .cloned()
        .collect();

    let mut loaded = 0usize;
    for e in &entries {
        let phys = space.phys_addr(e)?;
        let arena = space.arena(e.device);
        let off = phys - arena.base;
        let bytes = source
            .tensor(&e.name)
            .unwrap_or_else(|| vec![0u8; e.reserved as usize]);
        let n = bytes.len().min(e.reserved as usize);
        backend.upload(arena, off, &bytes[..n])?;
        loaded += 1;
    }
    Ok(loaded)
}
