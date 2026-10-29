//! Ahead-of-time compiled library kernels carried by a packet as cubins (no host library). The
//! object is a container (`scripts/fa4_aot/build.py`): `b"PLOWAOT1"`, a u32 manifest length, the
//! manifest JSON, then the cubins it indexes. Per kernel entry the manifest records the symbol, the
//! parameter layout read from the cubin (`(offset, size)` per ordinal), the cubin's sha256 and
//! exemplar launches captured from the exporter's own host code (grid, block, dynamic smem, launch
//! and function attributes, parameter bytes with never-written padding masked as `..`).
//!
//! plowrt launches the kernels itself: a host stub (named and versioned in the manifest) builds
//! every launch. Loading is fail-closed: the stub name, version and the pinned upstream must
//! match the runtime's, the driver's view of each kernel's parameters must equal the recorded
//! layout, and the stub must rebuild every exemplar byte for byte (a tensor map on its 64
//! defined bytes) before any entry is used.

use super::{CudaBackend, CudaStream, KernelFn};
use crate::device::{Backend, Module};
use crate::error::{Result, RuntimeError};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::c_void;

const MAGIC: &[u8; 8] = b"PLOWAOT1";
/// Bytes of a `CUtensorMap` the encoders define; the rest of the 128 are unused.
pub(crate) const TMAP_DEFINED: usize = 64;

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) format: u32,
    pub(crate) host_stub: String,
    pub(crate) host_stub_version: u32,
    pub(crate) upstream: BTreeMap<String, String>,
    pub(crate) abi: String,
    pub(crate) entries: Vec<Entry>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) role: String,
    pub(crate) kv_heads: u32,
    pub(crate) symbol: String,
    pub(crate) params: Vec<[u32; 2]>,
    pub(crate) sha256: String,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) exemplars: Vec<Exemplar>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Exemplar {
    pub(crate) args: Vec<i64>,
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) smem: u32,
    pub(crate) attrs: Vec<[i32; 2]>,
    pub(crate) kernel_attrs: Vec<[i32; 2]>,
    pub(crate) params: Vec<String>,
}

/// A parsed container. Blobs are checked against their recorded sha256 when taken.
pub(crate) struct Image {
    pub(crate) manifest: Manifest,
    bytes: Vec<u8>,
    base: usize,
}

impl Image {
    pub(crate) fn parse(bytes: Vec<u8>) -> Result<Image> {
        let bad = |why: &str| RuntimeError::Rejected(format!("AOT kernel object: {why}"));
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(bad("not a PLOWAOT1 container (re-emit the packet)"));
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
        let base = 12usize
            .checked_add(n)
            .filter(|&b| b <= bytes.len())
            .ok_or_else(|| bad("manifest past the end"))?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes[12..base]).map_err(|e| bad(&format!("manifest: {e}")))?;
        if manifest.format != 1 {
            return Err(bad(&format!("container format {}", manifest.format)));
        }
        for e in &manifest.entries {
            let end = e
                .offset
                .checked_add(e.bytes)
                .and_then(|end| end.checked_add(base as u64));
            if end.is_none_or(|end| end > bytes.len() as u64) {
                return Err(bad(&format!("entry {} past the end", e.name)));
            }
        }
        Ok(Image {
            manifest,
            bytes,
            base,
        })
    }

    pub(crate) fn entry(&self, name: &str) -> Result<&Entry> {
        self.manifest
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| RuntimeError::Rejected(format!("AOT kernel object: no entry {name}")))
    }

    pub(crate) fn blob(&self, e: &Entry) -> Result<&[u8]> {
        let at = self.base + e.offset as usize;
        let blob = &self.bytes[at..at + e.bytes as usize];
        let sha = Sha256::digest(blob)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if sha != e.sha256 {
            return Err(RuntimeError::Rejected(format!(
                "AOT kernel object: entry {} sha256 {sha} != recorded {}",
                e.name, e.sha256
            )));
        }
        Ok(blob)
    }
}

/// One loaded kernel with its parameter layout and the launch attributes its exemplars carry.
pub(crate) struct Kernel {
    pub(crate) f: KernelFn,
    pub(crate) module: Module,
    pub(crate) layout: Vec<(usize, usize)>,
    pub(crate) attrs: Vec<(i32, i32)>,
}

impl Kernel {
    /// Loads `e` from `image`; the driver's parameter layout must equal the recorded one and the
    /// exemplars' function attributes are applied.
    pub(crate) fn load(be: &CudaBackend, image: &Image, e: &Entry) -> Result<Kernel> {
        let module = be.module_load(image.blob(e)?)?;
        let f = be.get_function(&module, &e.symbol)?;
        let layout: Vec<(usize, usize)> = e
            .params
            .iter()
            .map(|&[o, s]| (o as usize, s as usize))
            .collect();
        let driver = be.param_layout(f)?.ok_or_else(|| {
            RuntimeError::Rejected(format!(
                "AOT kernel {}: the driver cannot report parameter layouts (needs CUDA 12.4+)",
                e.name
            ))
        })?;
        if driver != layout {
            return Err(RuntimeError::Rejected(format!(
                "AOT kernel {}: driver parameter layout {driver:?} != recorded {layout:?}",
                e.name
            )));
        }
        let first = e.exemplars.first().ok_or_else(|| {
            RuntimeError::Rejected(format!("AOT kernel {}: no exemplar launch", e.name))
        })?;
        if e.exemplars
            .iter()
            .any(|x| x.attrs != first.attrs || x.kernel_attrs != first.kernel_attrs)
        {
            return Err(RuntimeError::Rejected(format!(
                "AOT kernel {}: exemplars disagree on attributes",
                e.name
            )));
        }
        for &[attribute, value] in &first.kernel_attrs {
            be.set_function_attribute(f, attribute, value)?;
        }
        Ok(Kernel {
            f,
            module,
            layout,
            attrs: first.attrs.iter().map(|&[a, v]| (a, v)).collect(),
        })
    }

    pub(crate) fn launch(
        &self,
        be: &CudaBackend,
        l: &mut Launch,
        stream: &CudaStream,
    ) -> Result<()> {
        let base = l.params.as_mut_ptr() as *mut u8;
        let mut ptrs: smallvec::SmallVec<[*mut c_void; 32]> = self
            .layout
            .iter()
            // SAFETY: every offset lies inside the buffer (`Launch::new` sized it from the layout).
            .map(|&(o, _)| unsafe { base.add(o) } as *mut c_void)
            .collect();
        be.launch_ex(
            self.f,
            l.grid,
            l.block,
            l.smem,
            &self.attrs,
            &mut ptrs,
            stream,
        )
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(128))]
struct Block([u8; 128]);

/// One launch's configuration and parameter bytes (the buffer is 128-aligned, so a tensor map at
/// a 64-aligned parameter offset is 64-aligned).
pub(crate) struct Launch {
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) smem: u32,
    params: Vec<Block>,
    layout: Vec<(usize, usize)>,
    /// Bytes each parameter's builder wrote, and bytes it declared padding (never read by the
    /// kernel; a capture may hold stack residue there).
    written: Vec<Vec<bool>>,
    padding: Vec<Vec<bool>>,
    /// Parameters holding a tensor map.
    tmaps: Vec<bool>,
}

impl Launch {
    pub(crate) fn new(layout: &[(usize, usize)]) -> Launch {
        let total = layout.iter().map(|&(o, s)| o + s).max().unwrap_or(0);
        Launch {
            grid: [0; 3],
            block: [0; 3],
            smem: 0,
            params: vec![Block([0u8; 128]); total.div_ceil(128)],
            layout: layout.to_vec(),
            written: layout.iter().map(|&(_, s)| vec![false; s]).collect(),
            padding: layout.iter().map(|&(_, s)| vec![false; s]).collect(),
            tmaps: vec![false; layout.len()],
        }
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: [Block] is contiguous plain bytes (no padding: size 128, align 128).
        unsafe {
            std::slice::from_raw_parts_mut(
                self.params.as_mut_ptr() as *mut u8,
                self.params.len() * 128,
            )
        }
    }

    /// Writes `v` at byte `at` of parameter `index`.
    pub(crate) fn put(&mut self, index: usize, at: usize, v: &[u8]) {
        let (o, s) = self.layout[index];
        assert!(
            at + v.len() <= s,
            "parameter {index}: {} bytes at {at} past {s}",
            v.len()
        );
        self.bytes()[o + at..o + at + v.len()].copy_from_slice(v);
        self.written[index][at..at + v.len()].fill(true);
    }

    pub(crate) fn u32(&mut self, index: usize, at: usize, v: u32) {
        self.put(index, at, &v.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, index: usize, at: usize, v: u64) {
        self.put(index, at, &v.to_le_bytes());
    }

    /// Declares bytes `[at, at + len)` of parameter `index` padding.
    pub(crate) fn pad(&mut self, index: usize, at: usize, len: usize) {
        self.padding[index][at..at + len].fill(true);
    }

    /// A memref parameter: pointer, i32 dynamic extents, then (8-aligned) i64 dynamic strides;
    /// every other byte is padding.
    pub(crate) fn memref(&mut self, index: usize, ptr: u64, shapes: &[u32], strides: &[u64]) {
        self.u64(index, 0, ptr);
        self.extents(index, 8, shapes, strides);
    }

    /// A memref parameter whose dynamic strides are i32.
    pub(crate) fn memref32(&mut self, index: usize, ptr: u64, shapes: &[u32], strides: &[u32]) {
        self.u64(index, 0, ptr);
        let mut all = shapes.to_vec();
        all.extend_from_slice(strides);
        self.extents(index, 8, &all, &[]);
    }

    /// A coordinate tensor parameter: i32 dynamic extents, no pointer.
    pub(crate) fn coords(&mut self, index: usize, shapes: &[u32]) {
        self.extents(index, 0, shapes, &[]);
    }

    /// Writes the extents; the parameter's remaining bytes must be its 8-byte alignment tail
    /// (anything longer is a field this builder does not know: a layout change).
    fn extents(&mut self, index: usize, at: usize, shapes: &[u32], strides: &[u64]) {
        let mut o = at;
        for &v in shapes {
            self.u32(index, o, v);
            o += 4;
        }
        if !strides.is_empty() {
            let aligned = o.next_multiple_of(8);
            self.pad(index, o, aligned - o);
            o = aligned;
        }
        for &v in strides {
            self.u64(index, o, v);
            o += 8;
        }
        let size = self.layout[index].1;
        assert!(
            size >= o && size - o < 8,
            "parameter {index}: {size} bytes, the builder knows {o}"
        );
        self.pad(index, o, size - o);
    }

    pub(crate) fn tmap(&mut self, index: usize, map: &[u8; 128]) {
        self.put(index, 0, &map[..]);
        self.tmaps[index] = true;
    }

    /// Whether this launch reproduces `x`: config, and every captured byte the builder wrote, a
    /// tensor map on its defined bytes. A captured (non-padding) byte the builder left unwritten
    /// fails too, except a tensor map's undefined tail.
    pub(crate) fn matches(&mut self, x: &Exemplar) -> std::result::Result<(), String> {
        if (self.grid, self.block, self.smem) != (x.grid, x.block, x.smem) {
            return Err(format!(
                "config grid {:?} block {:?} smem {} != captured {:?} {:?} {}",
                self.grid, self.block, self.smem, x.grid, x.block, x.smem
            ));
        }
        if x.params.len() != self.layout.len() {
            return Err(format!(
                "{} captured parameters, {} in the layout",
                x.params.len(),
                self.layout.len()
            ));
        }
        let layout = self.layout.clone();
        for (i, hex) in x.params.iter().enumerate() {
            let (o, s) = layout[i];
            let defined = if self.tmaps[i] {
                TMAP_DEFINED.min(s)
            } else {
                s
            };
            let ours = self.bytes()[o..o + s].to_vec();
            for b in 0..defined {
                let cap = &hex
                    .get(2 * b..2 * b + 2)
                    .ok_or(format!("parameter {i}: short capture"))?;
                // A masked byte must be one the stub declares padding: a manifest cannot mask
                // a byte the kernel reads.
                if *cap == ".." {
                    if !(self.padding[i][b] && !self.written[i][b]) {
                        return Err(format!("parameter {i} byte {b}: masked, not padding"));
                    }
                    continue;
                }
                let want =
                    u8::from_str_radix(cap, 16).map_err(|e| format!("parameter {i}: {e}"))?;
                if self.padding[i][b] && !self.written[i][b] {
                    continue;
                }
                if !self.written[i][b] {
                    return Err(format!(
                        "parameter {i} byte {b}: captured {want:#04x}, not built"
                    ));
                }
                if ours[b] != want {
                    return Err(format!(
                        "parameter {i} byte {b}: built {:#04x}, captured {want:#04x}",
                        ours[b]
                    ));
                }
            }
        }
        Ok(())
    }
}
