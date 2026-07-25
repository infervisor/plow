//! §C Memory & weights — physical allocation, weight loading, KV paging, the
//! DMA plane (§M) and the tiered streamer (§N).

pub mod container;
pub mod dma;
pub mod kv;
pub mod pool;
pub mod prefix;
pub mod streamer;
pub mod tile_ref;
pub mod vmm;
pub mod weights;

use std::sync::Arc;

use plow_asset::{KvPaging, MemEntry, MemoryMap};

use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

/// Physical backing for a compiled address map: one arena per device segment,
/// plus the slot → physical-address rebase table.
///
/// "Allocation" at runtime is deliberately thin — one big arena per device (not
/// one per buffer), then a pure offset rebase. Aliases (overlapping `[offset,
/// reserved)` ranges) need no special handling; the map is pre-resolved.
pub struct AddressSpace {
    backend: Arc<dyn Backend>,
    /// One arena per device, indexed by device id.
    arenas: Vec<DeviceMem>,
    map: MemoryMap,
}

impl AddressSpace {
    /// Allocate every segment's arena and validate the map.
    pub fn allocate(backend: Arc<dyn Backend>, map: MemoryMap) -> Result<Self> {
        map.validate().map_err(RuntimeError::AddressMap)?;
        let max_device = map.segments.iter().map(|s| s.device).max().unwrap_or(0);
        let mut arenas: Vec<Option<DeviceMem>> = (0..=max_device).map(|_| None).collect();
        for seg in &map.segments {
            let mem = backend.alloc(seg.device, seg.size)?;
            arenas[seg.device as usize] = Some(mem);
        }
        let arenas = arenas
            .into_iter()
            .enumerate()
            .map(|(d, m)| {
                m.ok_or_else(|| RuntimeError::AddressMap(format!("no segment for device {d}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(AddressSpace {
            backend,
            arenas,
            map,
        })
    }

    pub fn map(&self) -> &MemoryMap {
        &self.map
    }

    /// Rebase a map entry to its physical device address:
    /// `arena_base(device) + (entry.offset − segment.global_base)`.
    pub fn phys_addr(&self, entry: &MemEntry) -> Result<u64> {
        let seg = self
            .map
            .segment_of(entry)
            .ok_or_else(|| RuntimeError::AddressMap(format!("no segment for '{}'", entry.name)))?;
        let local = entry.offset - seg.global_base;
        Ok(self.arenas[entry.device as usize].base + local)
    }

    /// Physical address of a named buffer's replica on `device`.
    pub fn addr_of(&self, name: &str, device: u8) -> Result<u64> {
        let entry = self
            .map
            .on_device(name, device)
            .ok_or_else(|| RuntimeError::AddressMap(format!("'{name}' not on device {device}")))?;
        self.phys_addr(entry)
    }

    /// The arena for `device` (for upload/download).
    pub fn arena(&self, device: u8) -> &DeviceMem {
        &self.arenas[device as usize]
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Resolve per-layer physical base addresses for the KV cache. Each entry
    /// is the `phys_addr` of the `kv_cache_L{i}` buffer the compiler emitted
    /// (matched by `KvLayerPaging::buffer_name`). Missing entries default to
    /// zero so the runtime can still construct a `KvArena` for a partial map
    /// (test bundles that don't declare the layers), and OOB layers return
    /// zero rather than panicking — the mux logs a warning at spawn if any
    /// layer is unmapped.
    pub fn kv_layer_bases(&self, paging: &KvPaging) -> Vec<u64> {
        paging
            .per_layer
            .iter()
            .map(|lp| {
                self.map
                    .get(&lp.buffer_name)
                    .and_then(|entry| self.phys_addr(entry).ok())
                    .unwrap_or(0)
            })
            .collect()
    }
}
