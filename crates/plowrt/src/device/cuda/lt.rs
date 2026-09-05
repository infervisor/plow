use std::ffi::c_void;
use std::sync::Arc;

use super::{CudaBackend, CudaStream};
use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

type Handle = *mut c_void;
type Status = i32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Algo {
    data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Heuristic {
    algo: Algo,
    workspace: usize,
    state: Status,
    waves: f32,
    reserved: [i32; 4],
}

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) -> Status),+ $(,)?) => {
        #[allow(non_snake_case)]
        struct Api {
            _lib: libloading::Library,
            $($name: unsafe extern "C" fn($($arg),*) -> Status,)+
        }
        impl Api {
            #[allow(non_snake_case)]
            fn load() -> Result<Self> {
                let mut last = String::new();
                for path in ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"] {
                    // SAFETY: optional NVIDIA host library, retained with its symbols.
                    let lib = match unsafe { libloading::Library::new(path) } {
                        Ok(lib) => lib,
                        Err(e) => { last = e.to_string(); continue; }
                    };
                    $(let $name = *unsafe {
                        lib.get::<unsafe extern "C" fn($($arg),*) -> Status>(
                            concat!(stringify!($name), "\0").as_bytes())
                    }.map_err(|e| RuntimeError::Device(format!("resolve {}: {e}", stringify!($name))))?;)+
                    return Ok(Self { _lib: lib, $($name,)+ });
                }
                Err(RuntimeError::Device(format!("load cuBLASLt: {last}")))
            }
        }
    };
}

api! {
    cublasLtCreate: fn(*mut Handle) -> Status,
    cublasLtDestroy: fn(Handle) -> Status,
    cublasLtMatmulDescCreate: fn(*mut Handle, i32, i32) -> Status,
    cublasLtMatmulDescDestroy: fn(Handle) -> Status,
    cublasLtMatmulDescSetAttribute: fn(Handle, i32, *const c_void, usize) -> Status,
    cublasLtMatrixLayoutCreate: fn(*mut Handle, i32, u64, u64, i64) -> Status,
    cublasLtMatrixLayoutDestroy: fn(Handle) -> Status,
    cublasLtMatmulPreferenceCreate: fn(*mut Handle) -> Status,
    cublasLtMatmulPreferenceDestroy: fn(Handle) -> Status,
    cublasLtMatmulPreferenceSetAttribute: fn(Handle, i32, *const c_void, usize) -> Status,
    cublasLtMatmulAlgoGetHeuristic: fn(Handle, Handle, Handle, Handle, Handle, Handle, Handle, i32, *mut Heuristic, *mut i32) -> Status,
    cublasLtMatmul: fn(Handle, Handle, *const c_void, *const c_void, Handle, *const c_void, Handle, *const c_void, *const c_void, Handle, *mut c_void, Handle, *const Algo, *mut c_void, usize, Handle) -> Status,
}

fn check(status: Status, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(RuntimeError::Device(format!(
            "{op}: cuBLASLt status {status}"
        )))
    }
}

pub(crate) struct Lt {
    be: Arc<CudaBackend>,
    api: Api,
    handle: usize,
    workspace: DeviceMem,
}

impl Lt {
    pub(crate) fn load(be: &Arc<CudaBackend>) -> Result<Arc<Self>> {
        be.bind()?;
        let api = Api::load()?;
        let workspace = be.alloc(0, 256 * 1024 * 1024)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: output handle valid; the library and CUDA context outlive it.
        unsafe {
            check((api.cublasLtCreate)(&mut handle), "cublasLtCreate")?;
        }
        Ok(Arc::new(Self {
            be: be.clone(),
            api,
            handle: handle as usize,
            workspace,
        }))
    }

    pub(crate) fn plan(self: &Arc<Self>, m: u32, n: u32, k: u32, weight: u64) -> Result<Arc<Plan>> {
        self.be.bind()?;
        let mut plan = Plan {
            lt: self.clone(),
            desc: 0,
            w: 0,
            a: 0,
            c: 0,
            algo: Algo::default(),
        };
        let mut raw = std::ptr::null_mut();
        // CUDA headers: BF16=14, FP32=0, COMPUTE_32F=68, TRANSA attribute=3, OP_T=1.
        // SAFETY: exact CUDA 12/13 C ABI; Plan drops any descriptors created before an error.
        unsafe {
            check(
                (self.api.cublasLtMatmulDescCreate)(&mut raw, 68, 0),
                "Lt descriptor",
            )?;
            plan.desc = raw as usize;
            let trans = 1i32;
            check(
                (self.api.cublasLtMatmulDescSetAttribute)(
                    raw,
                    3,
                    &trans as *const _ as *const c_void,
                    size_of::<i32>(),
                ),
                "Lt transpose",
            )?;
            for (dst, rows, cols, ld) in [
                (&mut plan.w, k as u64, n as u64, k as i64),
                (&mut plan.a, k as u64, m as u64, k as i64),
                (&mut plan.c, n as u64, m as u64, n as i64),
            ] {
                raw = std::ptr::null_mut();
                check(
                    (self.api.cublasLtMatrixLayoutCreate)(&mut raw, 14, rows, cols, ld),
                    "Lt layout",
                )?;
                *dst = raw as usize;
            }
            let mut pref = std::ptr::null_mut();
            check(
                (self.api.cublasLtMatmulPreferenceCreate)(&mut pref),
                "Lt preference",
            )?;
            let result = (|| {
                let bytes = self.workspace.len as usize;
                check(
                    (self.api.cublasLtMatmulPreferenceSetAttribute)(
                        pref,
                        1,
                        &bytes as *const _ as *const c_void,
                        size_of::<usize>(),
                    ),
                    "Lt workspace",
                )?;
                let mut results = [Heuristic::default(); 8];
                let mut count = 0;
                check(
                    (self.api.cublasLtMatmulAlgoGetHeuristic)(
                        self.handle as Handle,
                        plan.desc as Handle,
                        plan.w as Handle,
                        plan.a as Handle,
                        plan.c as Handle,
                        plan.c as Handle,
                        pref,
                        results.len() as i32,
                        results.as_mut_ptr(),
                        &mut count,
                    ),
                    "Lt heuristic",
                )?;
                let winner = results
                    .get(..count as usize)
                    .unwrap_or(&[])
                    .iter()
                    .find(|r| r.state == 0 && r.workspace <= bytes)
                    .ok_or_else(|| {
                        RuntimeError::Device("no supported BF16 cuBLASLt algorithm".into())
                    })?;
                plan.algo = winner.algo;
                plan.select(
                    results.get(..count as usize).unwrap_or(&[]),
                    m,
                    n,
                    k,
                    weight,
                )?;
                Ok(())
            })();
            (self.api.cublasLtMatmulPreferenceDestroy)(pref);
            result?;
        }
        Ok(Arc::new(plan))
    }
}

impl Drop for Lt {
    fn drop(&mut self) {
        if self.be.bind().is_ok() {
            // SAFETY: engine unload synchronizes before its cached plans are dropped.
            unsafe {
                (self.api.cublasLtDestroy)(self.handle as Handle);
            }
        }
    }
}

pub(crate) struct Plan {
    lt: Arc<Lt>,
    desc: usize,
    w: usize,
    a: usize,
    c: usize,
    algo: Algo,
}

impl Plan {
    fn select(
        &mut self,
        candidates: &[Heuristic],
        m: u32,
        n: u32,
        k: u32,
        weight: u64,
    ) -> Result<()> {
        let be = Arc::clone(&self.lt.be);
        let bytes_w = n as u64 * k as u64 * 2;
        let repeats = ((700 * 1024 * 1024u64).div_ceil(bytes_w)).clamp(2, 16);
        let copies = be.alloc(0, repeats * bytes_w)?;
        for i in 0..repeats {
            be.memcpy_dtod(copies.base + i * bytes_w, weight, bytes_w)?;
        }
        let input = be.alloc(0, m as u64 * k as u64 * 2)?;
        let output = be.alloc(0, m as u64 * n as u64 * 2)?;
        let stream = be.stream_create()?;
        be.memset_d8_async(input.base, 0x3c, input.len as usize, &stream)?;
        let start = be.event_create(true)?;
        let end = be.event_create(true)?;
        let mut best = f32::INFINITY;
        let mut selected = None;
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.state != 0 || candidate.workspace > self.lt.workspace.len as usize {
                continue;
            }
            self.algo = candidate.algo;
            let run = |i: u64| {
                self.run(
                    input.base,
                    copies.base + i % repeats * bytes_w,
                    output.base,
                    &stream,
                )
            };
            // Complete a cold weight ring for every candidate; event timing excludes allocation/copy.
            let result = (|| {
                for i in 0..repeats {
                    run(i)?;
                }
                be.event_record(&start, &stream)?;
                for i in 0..repeats * 2 {
                    run(i)?;
                }
                be.event_record(&end, &stream)?;
                be.event_synchronize(&end)?;
                be.event_elapsed_ms(&start, &end)
            })();
            match result {
                Ok(ms) if ms < best => {
                    best = ms;
                    selected = Some(index);
                }
                Ok(_) => {}
                Err(e) => {
                    // Drain before trying another algorithm or releasing its scratch.
                    be.stream_synchronize(&stream)?;
                    if e.is_fatal() {
                        return Err(e);
                    }
                    tracing::warn!(error = %e, index, "cuBLASLt candidate rejected");
                }
            }
        }
        be.stream_synchronize(&stream)?;
        let index = selected
            .ok_or_else(|| RuntimeError::Device("no runnable cuBLASLt candidate".into()))?;
        self.algo = candidates[index].algo;
        tracing::info!(
            m,
            n,
            k,
            index,
            candidates = candidates.len(),
            workspace = candidates[index].workspace,
            cold_bytes = copies.len,
            matmul_ms = best / (repeats * 2) as f32,
            "cuBLASLt load-time algorithm selected"
        );
        Ok(())
    }

    pub(crate) fn run(&self, a: u64, w: u64, c: u64, stream: &CudaStream) -> Result<()> {
        self.lt.be.bind()?;
        let alpha = 1f32;
        let beta = 0f32;
        // SAFETY: route validation established BF16 extents and nonaliasing at load.
        // All work is serialized on the engine stream, including shared workspace use.
        unsafe {
            check(
                (self.lt.api.cublasLtMatmul)(
                    self.lt.handle as Handle,
                    self.desc as Handle,
                    &alpha as *const _ as *const c_void,
                    w as *const c_void,
                    self.w as Handle,
                    a as *const c_void,
                    self.a as Handle,
                    &beta as *const _ as *const c_void,
                    c as *const c_void,
                    self.c as Handle,
                    c as *mut c_void,
                    self.c as Handle,
                    &self.algo,
                    self.lt.workspace.base as *mut c_void,
                    self.lt.workspace.len as usize,
                    stream.raw as Handle,
                ),
                "cublasLtMatmul",
            )
        }
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if self.lt.be.bind().is_ok() {
            // SAFETY: each nonzero handle was created once and belongs to this plan.
            unsafe {
                for layout in [self.w, self.a, self.c] {
                    if layout != 0 {
                        (self.lt.api.cublasLtMatrixLayoutDestroy)(layout as Handle);
                    }
                }
                if self.desc != 0 {
                    (self.lt.api.cublasLtMatmulDescDestroy)(self.desc as Handle);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_header_layout() {
        assert_eq!(size_of::<Algo>(), 64);
        assert_eq!(align_of::<Algo>(), 8);
        assert_eq!(size_of::<Heuristic>(), 96);
        assert_eq!(std::mem::offset_of!(Heuristic, workspace), 64);
        assert_eq!(std::mem::offset_of!(Heuristic, state), 72);
        assert_eq!(std::mem::offset_of!(Heuristic, waves), 76);
        assert_eq!(std::mem::offset_of!(Heuristic, reserved), 80);
    }
}
