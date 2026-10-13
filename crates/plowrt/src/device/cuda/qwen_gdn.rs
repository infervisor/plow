use super::{CudaBackend, CudaStream};
use crate::{
    device::{Backend, DeviceMem},
    Result, RuntimeError,
};
use std::{ffi::c_void, path::Path, sync::Arc};

const MAX_TOKENS: usize = 8192;
const MAP_BYTES: u64 = 16896;
/// Per-sequence recurrent state `[HV=48, V=128, K=128]` f32, the library's fixed geometry.
pub(crate) const STATE_BYTES: u64 = 48 * 128 * 128 * 4;
/// Bytes per row of the conv core tensors (`validate_n` sizes 2 and 3).
pub(crate) const CORE_ROW_BYTES: u64 = 12288;
/// Packed sequences one launch takes (cu_seqlens entries - 1).
pub(crate) const MAX_SEQS: usize = 64;
type Create = unsafe extern "C" fn(i32, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Run = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut c_void,
) -> i32;
type RunVarlen = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    i32,
    *mut c_void,
) -> i32;

pub(crate) struct NativeGdn {
    _library: libloading::Library,
    handle: usize,
    destroy: Destroy,
    run: Run,
    /// `plow_gdn_run_varlen`: libraries built before the packed chain lack it.
    run_varlen: Option<RunVarlen>,
    maps: DeviceMem,
    offsets: DeviceMem,
    /// cu_seqlens of the current packed launch (device, MAX_SEQS + 1 i64) and its host image,
    /// which the async upload reads until the caller's stream synchronize.
    cu: DeviceMem,
    cu_host: Vec<i64>,
    stream: Option<usize>,
    backend: Arc<CudaBackend>,
}

fn check(rc: i32, operation: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else if rc == -1001 {
        Err(RuntimeError::Device("native GDN library already has a live handle; one engine per loaded library is supported".into()))
    } else {
        Err(RuntimeError::Device(format!(
            "{operation}: native GDN status {rc}"
        )))
    }
}

impl NativeGdn {
    pub(crate) fn load(backend: Arc<CudaBackend>, path: &Path) -> Result<Self> {
        backend.bind()?;
        // The configured library implements the fixed native GDN ABI.
        let library = unsafe { libloading::Library::new(path) }.map_err(|e| {
            RuntimeError::Device(format!("load native GDN {}: {e}", path.display()))
        })?;
        let create = *unsafe { library.get::<Create>(b"plow_gdn_create\0") }
            .map_err(|e| RuntimeError::Device(format!("resolve plow_gdn_create: {e}")))?;
        let destroy = *unsafe { library.get::<Destroy>(b"plow_gdn_destroy\0") }
            .map_err(|e| RuntimeError::Device(format!("resolve plow_gdn_destroy: {e}")))?;
        let run = *unsafe { library.get::<Run>(b"plow_gdn_run\0") }
            .map_err(|e| RuntimeError::Device(format!("resolve plow_gdn_run: {e}")))?;
        let run_varlen = unsafe { library.get::<RunVarlen>(b"plow_gdn_run_varlen\0") }
            .ok()
            .map(|f| *f);
        let cu = backend.alloc(backend.device_ordinal, ((MAX_SEQS + 1) * 8) as u64)?;
        let maps = backend.alloc(backend.device_ordinal, MAP_BYTES)?;
        if maps.base % 128 != 0 {
            return Err(RuntimeError::Device(
                "native GDN maps require 128-byte alignment".into(),
            ));
        }
        let offsets = backend.alloc(backend.device_ordinal, (MAX_TOKENS * 16) as u64)?;
        let mut pairs = Vec::with_capacity(MAX_TOKENS * 16);
        for tokens in 1..=MAX_TOKENS {
            pairs.extend_from_slice(&0_i64.to_le_bytes());
            pairs.extend_from_slice(&(tokens as i64).to_le_bytes());
        }
        backend.memcpy_htod(offsets.base, &pairs)?;
        let mut handle = std::ptr::null_mut();
        check(
            unsafe { create(backend.device_ordinal as i32, &mut handle) },
            "plow_gdn_create",
        )?;
        if handle.is_null() {
            return Err(RuntimeError::Device(
                "native GDN returned a null handle".into(),
            ));
        }
        Ok(Self {
            _library: library,
            handle: handle as usize,
            destroy,
            run,
            run_varlen,
            maps,
            offsets,
            cu,
            cu_host: Vec::with_capacity(MAX_SEQS + 1),
            stream: None,
            backend,
        })
    }

    /// Enqueues BF16 Q/K/V/output, F32 alpha/beta, and F32 V-first states.
    /// The native wrapper copies output state back to initial state on this stream.
    ///
    /// # Safety
    /// Views must remain live in this context with the stated layouts until stream
    /// completion. No other stream may race writable tensors. Keep this stream
    /// alive until the adapter is dropped.
    pub(crate) unsafe fn launch(
        &mut self,
        tensors: [&DeviceMem; 8],
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        validate(&tensors, tokens)?;
        if stream.keep.ctx != self.backend.ctx || self.stream.is_some_and(|s| s != stream.raw) {
            return Err(RuntimeError::Device(
                "native GDN requires one stream in its selected context".into(),
            ));
        }
        self.backend.bind()?;
        self.stream = Some(stream.raw);
        let p = tensors.map(|t| t.base as *mut c_void);
        let offsets = self.offsets.base + ((tokens - 1) * 16) as u64;
        // Bounds and aliases are checked above; caller owns async lifetimes.
        check(
            unsafe {
                (self.run)(
                    self.handle as *mut c_void,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    p[5],
                    p[6],
                    p[7],
                    self.maps.base as *mut c_void,
                    offsets as *mut c_void,
                    tokens as i32,
                    stream.raw as *mut c_void,
                )
            },
            "plow_gdn_run",
        )
    }
}

fn validate(tensors: &[&DeviceMem; 8], tokens: usize) -> Result<()> {
    validate_n(tensors, tokens, 1)
}

/// `tensors[6]` / `[7]` hold `nseq` contiguous state slots.
fn validate_n(tensors: &[&DeviceMem; 8], tokens: usize, nseq: usize) -> Result<()> {
    if !(1..=MAX_TOKENS).contains(&tokens) {
        return Err(RuntimeError::Device(
            "native GDN tokens must be in 1..=8192".into(),
        ));
    }
    if !(1..=MAX_SEQS).contains(&nseq) || nseq > tokens {
        return Err(RuntimeError::Device(
            "native GDN packed sequences must be in 1..=64 and non-empty".into(),
        ));
    }
    let t = tokens as u64;
    let sizes = [
        t * 4096,
        t * 4096,
        t * 12288,
        t * 12288,
        t * 192,
        t * 192,
        nseq as u64 * STATE_BYTES,
        nseq as u64 * STATE_BYTES,
    ];
    let mut ends = [0; 8];
    for (i, tensor) in tensors.iter().enumerate() {
        let alignment = if i == 4 || i == 5 { 4 } else { 16 };
        if tensor.base == 0 || tensor.base % alignment != 0 || tensor.len < sizes[i] {
            return Err(RuntimeError::Device(format!(
                "native GDN tensor {i}: invalid extent/alignment"
            )));
        }
        ends[i] = tensor
            .base
            .checked_add(sizes[i])
            .ok_or_else(|| RuntimeError::Device("native GDN address overflow".into()))?;
    }
    for i in [3, 6, 7] {
        for j in 0..8 {
            if i != j && tensors[i].base < ends[j] && tensors[j].base < ends[i] {
                return Err(RuntimeError::Device(
                    "native GDN writable tensor aliases another operand".into(),
                ));
            }
        }
    }
    Ok(())
}

impl NativeGdn {
    pub(crate) fn varlen(&self) -> bool {
        self.run_varlen.is_some()
    }

    /// `lens.len()` packed sequences, rows contiguous in `lens` order; `tensors[6]` / `[7]` are
    /// `lens.len()` contiguous final / initial state slots (the caller gathers and scatters the
    /// per-slot states; nothing is copied back here).
    ///
    /// # Safety
    /// As [`Self::launch`]; the caller synchronizes the stream before the next packed launch
    /// (the cu_seqlens upload reads a host buffer owned by the adapter).
    pub(crate) unsafe fn launch_varlen(
        &mut self,
        tensors: [&DeviceMem; 8],
        lens: &[u32],
        stream: &CudaStream,
    ) -> Result<()> {
        let run = self.run_varlen.ok_or_else(|| {
            RuntimeError::Device("native GDN library lacks plow_gdn_run_varlen".into())
        })?;
        if lens.iter().any(|&l| l == 0) {
            return Err(RuntimeError::Device(
                "native GDN packed sequence of 0 rows".into(),
            ));
        }
        let tokens: usize = lens.iter().map(|&l| l as usize).sum();
        validate_n(&tensors, tokens, lens.len())?;
        if stream.keep.ctx != self.backend.ctx || self.stream.is_some_and(|s| s != stream.raw) {
            return Err(RuntimeError::Device(
                "native GDN requires one stream in its selected context".into(),
            ));
        }
        self.backend.bind()?;
        self.stream = Some(stream.raw);
        self.cu_host.clear();
        self.cu_host.push(0);
        let mut end = 0i64;
        for &l in lens {
            end += i64::from(l);
            self.cu_host.push(end);
        }
        // SAFETY: cu_host lives on self until the caller's stream synchronize.
        unsafe {
            self.backend.memcpy_htod_async(
                self.cu.base,
                bytemuck::cast_slice(&self.cu_host),
                stream,
            )?;
        }
        let p = tensors.map(|t| t.base as *mut c_void);
        check(
            unsafe {
                run(
                    self.handle as *mut c_void,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    p[5],
                    p[6],
                    p[7],
                    self.maps.base as *mut c_void,
                    self.cu.base as *mut c_void,
                    tokens as i32,
                    lens.len() as i32,
                    stream.raw as *mut c_void,
                )
            },
            "plow_gdn_run_varlen",
        )
    }
}

impl Drop for NativeGdn {
    fn drop(&mut self) {
        if self.backend.bind().is_ok() {
            // Destruction is cold; retire queued work before unloading code.
            let _ = self.backend.synchronize();
            let rc = unsafe { (self.destroy)(self.handle as *mut c_void) };
            if rc != 0 {
                tracing::debug!(rc, "plow_gdn_destroy failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensors() -> [DeviceMem; 8] {
        std::array::from_fn(|i| DeviceMem::view((i as u64 + 1) << 30, 128 << 20))
    }

    #[test]
    fn accepts_token_endpoints_and_readonly_aliases() {
        let mut t = tensors();
        assert!(validate(&t.each_ref(), 1).is_ok());
        assert!(validate(&t.each_ref(), MAX_TOKENS).is_ok());
        t[1] = DeviceMem::view(t[0].base, t[0].len);
        assert!(validate(&t.each_ref(), MAX_TOKENS).is_ok());
    }

    #[test]
    fn rejects_invalid_launch_ranges_before_ffi() {
        let mut t = tensors();
        assert!(validate(&t.each_ref(), 0).is_err());
        assert!(validate(&t.each_ref(), MAX_TOKENS + 1).is_err());
        t[2].len = 12287;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[0].base += 2;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[7].base = t[6].base;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[3].base = t[0].base + 16;
        assert!(validate(&t.each_ref(), 1).is_err());
        t = tensors();
        t[0].base = u64::MAX - 15;
        assert!(validate(&t.each_ref(), 1).is_err());
    }
}
