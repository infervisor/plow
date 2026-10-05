//! Small-M FP8 W8A8 decode projections on CUTLASS sm90 (`runtime/nvidia/cutlass_fp8_decode_sm90.cu`):
//! a cubin of `plow_cutlass_fp8_<cfg>` kernels and the host library that marshals their Params
//! blob (TMA descriptors included) for fixed operand addresses. Each routed segment owns one blob.
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use super::{CudaBackend, CudaStream, KernelFn};
use crate::device::{Backend, Module};
use crate::{Result, RuntimeError};

pub(crate) const CUBIN: &str = "cutlass_fp8_decode_sm90.cubin";
pub(crate) const HOST_LIB: &str = "libplow_cutlass_fp8_decode_sm90.so";
const CONFIGS: usize = 6;
const PARAMS_CAP: usize = 4096;

type PrepareFn = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    *mut c_void,
    *const f32,
    *const f32,
    *mut c_void,
    u32,
    *mut u32,
) -> i32;

pub(crate) struct CutlassFp8 {
    be: Arc<CudaBackend>,
    module: Module,
    functions: [KernelFn; CONFIGS],
    params_bytes: [u32; CONFIGS],
    prepare: PrepareFn,
    _lib: libloading::Library,
}

impl Drop for CutlassFp8 {
    fn drop(&mut self) {
        if let Err(error) = self.be.module_unload(&self.module) {
            tracing::warn!(%error, "unload CUTLASS FP8 decode object failed");
        }
    }
}

pub(crate) struct Plan {
    lib: Arc<CutlassFp8>,
    function: KernelFn,
    params: Vec<u8>,
    grid: [u32; 3],
    block: u32,
    smem: u32,
}

impl CutlassFp8 {
    /// `pins` = `<cubin sha256>:<host library sha256>`.
    pub(crate) fn load(be: &Arc<CudaBackend>, dir: &Path, pins: &str) -> Result<Arc<Self>> {
        let reject = |why: String| RuntimeError::Rejected(format!("CUTLASS FP8 decode: {why}"));
        let (cubin_pin, host_pin) =
            pins.split_once(':').ok_or_else(|| reject("expected <cubin sha256>:<host sha256>".into()))?;
        let cubin = dir.join(CUBIN);
        let image = std::fs::read(&cubin).map_err(|e| reject(format!("{}: {e}", cubin.display())))?;
        let host = dir.join(HOST_LIB);
        let host_image = std::fs::read(&host).map_err(|e| reject(format!("{}: {e}", host.display())))?;
        for (path, bytes, pin) in [(&cubin, &image, cubin_pin), (&host, &host_image, host_pin)] {
            if plow_asset::decode_objects::image_sha256(bytes) != pin {
                return Err(reject(format!("{} does not match its pinned sha256", path.display())));
            }
        }
        // SAFETY: the host library is a plain C ABI marshaller built from the same source.
        let lib = unsafe { libloading::Library::new(&host) }.map_err(|e| reject(format!("{}: {e}", host.display())))?;
        // SAFETY: symbol type matches `plow_cutlass_fp8_prepare`.
        let prepare = *unsafe { lib.get::<PrepareFn>(b"plow_cutlass_fp8_prepare\0") }
            .map_err(|e| reject(format!("resolve prepare: {e}")))?;
        be.bind()?;
        let module = be.module_load(&image)?;
        let lib = Self {
            be: Arc::clone(be),
            functions: [KernelFn(0); CONFIGS],
            params_bytes: [0; CONFIGS],
            module,
            prepare,
            _lib: lib,
        };
        if be.module_global_u32(&lib.module, "plow_cutlass_fp8_abi")? != Some(1) {
            return Err(reject("cubin ABI".into()));
        }
        let mut lib = lib;
        for cfg in 0..CONFIGS {
            lib.functions[cfg] = be.get_function(&lib.module, &format!("plow_cutlass_fp8_{cfg}"))?;
            lib.params_bytes[cfg] = be
                .module_global_u32(&lib.module, &format!("plow_cutlass_fp8_params_bytes_{cfg}"))?
                .ok_or_else(|| reject(format!("cfg {cfg} params size missing")))?;
        }
        tracing::info!(dir = %dir.display(), pins, "CUTLASS FP8 decode objects loaded");
        Ok(Arc::new(lib))
    }

    /// `None` when no configuration covers `m`. `x`/`sx` are the FP8 activations and their
    /// per-token scales, `w`/`sw` the weights and per-channel scales, `d` the BF16 output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan(
        self: &Arc<Self>,
        m: u32,
        n: u32,
        k: u32,
        [x, w, d, sx, sw]: [u64; 5],
        fast_accum: bool,
    ) -> Result<Option<Plan>> {
        let class = match m {
            1..=32 => 0,
            33..=64 => 1,
            65..=128 => 2,
            _ => return Ok(None),
        };
        let cfg = class + if fast_accum { 0 } else { 3 };
        let mut params = vec![0u8; PARAMS_CAP];
        let mut geom = [0u32; 5];
        // SAFETY: the marshaller writes at most PARAMS_CAP bytes and five geometry words; the
        // device addresses are only encoded, never dereferenced on the host.
        let bytes = unsafe {
            (self.prepare)(
                cfg as i32,
                m as i32,
                n as i32,
                k as i32,
                x as *const c_void,
                w as *const c_void,
                d as *mut c_void,
                sx as *const f32,
                sw as *const f32,
                params.as_mut_ptr().cast(),
                PARAMS_CAP as u32,
                geom.as_mut_ptr(),
            )
        };
        if bytes <= 0 {
            return Err(RuntimeError::Rejected(format!(
                "CUTLASS FP8 decode cfg {cfg} rejects m={m} n={n} k={k} ({bytes})"
            )));
        }
        if bytes as u32 != self.params_bytes[cfg] {
            return Err(RuntimeError::Rejected(
                "CUTLASS FP8 decode host library and cubin differ".into(),
            ));
        }
        params.truncate(bytes as usize);
        let function = self.functions[cfg];
        self.be.set_max_dynamic_smem(function, geom[4])?;
        Ok(Some(Plan {
            lib: Arc::clone(self),
            function,
            params,
            grid: [geom[0], geom[1], geom[2]],
            block: geom[3],
            smem: geom[4],
        }))
    }
}

impl Plan {
    pub(crate) fn run(&self, stream: &CudaStream) -> Result<()> {
        let mut args = [self.params.as_ptr() as *mut c_void];
        self.lib
            .be
            .launch_kernel_grid(self.function, self.grid, self.block, self.smem, &mut args, Some(stream))
    }
}
