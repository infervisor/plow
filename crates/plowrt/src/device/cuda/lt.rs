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
    ($($name:ident: fn($($arg:ty),*) -> Status),+ ;
     optional $($oname:ident: fn($($oarg:ty),*) -> Status),+ $(,)?) => {
        #[allow(non_snake_case)]
        struct Api {
            _lib: libloading::Library,
            $($name: unsafe extern "C" fn($($arg),*) -> Status,)+
            $($oname: Option<unsafe extern "C" fn($($oarg),*) -> Status>,)+
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
                    $(let $oname = unsafe {
                        lib.get::<unsafe extern "C" fn($($oarg),*) -> Status>(
                            concat!(stringify!($oname), "\0").as_bytes())
                    }.ok().map(|symbol| *symbol);)+
                    return Ok(Self { _lib: lib, $($name,)+ $($oname,)+ });
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
    cublasLtMatmulAlgoCheck: fn(Handle, Handle, Handle, Handle, Handle, Handle, *const Algo, *mut Heuristic) -> Status,
    cublasLtMatmul: fn(Handle, Handle, *const c_void, *const c_void, Handle, *const c_void, Handle, *const c_void, *const c_void, Handle, *mut c_void, Handle, *const Algo, *mut c_void, usize, Handle) -> Status;
    // Experimental in cuBLASLt 13.x (Hopper from 13.4): group shapes and matrix pointers are
    // DEVICE arrays, so a grouped matmul needs no host-side sizes.
    optional cublasLtGroupedMatrixLayoutCreate: fn(*mut Handle, i32, i32, *const c_void, *const c_void, *const c_void) -> Status,
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

/// One line of the per-shape algorithm table (`PLOW_LT_ALGOS`, JSONL). The algorithm is the
/// opaque 64-byte `cublasLtMatmulAlgo_t` the load-time selection picked for exactly this
/// `(m, n, k)` BF16 shape on this GPU; the runtime re-validates it with `cublasLtMatmulAlgoCheck`
/// before use, so a table from another GPU or library version degrades to the heuristic path,
/// never to a wrong launch. Written by the same runtime under `PLOW_LT_ALGOS_WRITE`, so a
/// build stage with a GPU can produce it once and every later load (or a GPU-less emit through
/// tunedb) reuses it instead of re-timing eight candidates per shape at serve start.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub(crate) struct StoredAlgo {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: String,
    pub gpu: String,
    /// The eight 64-bit words of `cublasLtMatmulAlgo_t`, hex.
    pub algo: [String; 8],
    pub workspace: usize,
    pub matmul_us: f32,
}

impl StoredAlgo {
    fn to_algo(&self) -> Option<Algo> {
        let mut data = [0u64; 8];
        for (dst, src) in data.iter_mut().zip(&self.algo) {
            *dst = u64::from_str_radix(src, 16).ok()?;
        }
        Some(Algo { data })
    }
}

pub(crate) struct Lt {
    be: Arc<CudaBackend>,
    api: Api,
    handle: usize,
    workspace: DeviceMem,
    /// `PLOW_LT_ALGOS`: shapes whose algorithm is pinned by the table.
    stored: std::collections::HashMap<(u32, u32, u32), Algo>,
    /// `PLOW_LT_ALGOS_WRITE`: append every load-time selection here.
    write: Option<std::path::PathBuf>,
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
        let nv = &crate::config::RuntimeConfig::get().nv;
        let mut stored = std::collections::HashMap::new();
        if let Some(path) = nv.lt_algos.as_deref().map(std::path::PathBuf::from) {
            let text = std::fs::read_to_string(&path).map_err(|e| {
                RuntimeError::Device(format!("PLOW_LT_ALGOS {}: {e}", path.display()))
            })?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let rec: StoredAlgo = serde_json::from_str(line).map_err(|e| {
                    RuntimeError::Device(format!("PLOW_LT_ALGOS: bad record: {e}"))
                })?;
                if rec.dtype != "bf16" {
                    continue;
                }
                if let Some(algo) = rec.to_algo() {
                    stored.insert((rec.m, rec.n, rec.k), algo);
                }
            }
            tracing::info!(
                path = %path.display(),
                shapes = stored.len(),
                "cuBLASLt algorithm table loaded"
            );
        }
        let write = nv.lt_algos_write.as_deref().map(std::path::PathBuf::from);
        Ok(Arc::new(Self {
            be: be.clone(),
            api,
            handle: handle as usize,
            workspace,
            stored,
            write,
        }))
    }

    fn record(&self, m: u32, n: u32, k: u32, algo: &Algo, workspace: usize, matmul_us: f32) {
        let Some(path) = &self.write else { return };
        let rec = StoredAlgo {
            m,
            n,
            k,
            dtype: "bf16".into(),
            gpu: self.be.device_name().to_string(),
            algo: std::array::from_fn(|i| format!("{:016x}", algo.data[i])),
            workspace,
            matmul_us,
        };
        let line = match serde_json::to_string(&rec) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "cuBLASLt algorithm record not serialized");
                return;
            }
        };
        use std::io::Write as _;
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| writeln!(f, "{line}"));
        if let Err(e) = result {
            tracing::warn!(error = %e, path = %path.display(), "cuBLASLt algorithm record not written");
        }
    }

    pub(crate) fn plan(
        self: &Arc<Self>,
        m: u32,
        n: u32,
        k: u32,
        weight: u64,
        template: Option<&Plan>,
    ) -> Result<Arc<Plan>> {
        if template.is_some_and(|p| {
            !Arc::ptr_eq(self, &p.lt) || p.shape.0 < m || (p.shape.1, p.shape.2) != (n, k)
        }) {
            return Err(RuntimeError::Rejected(
                "cuBLASLt rung template geometry differs".into(),
            ));
        }
        self.be.bind()?;
        let mut plan = Plan {
            lt: self.clone(),
            shape: (m, n, k),
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
            // A rung template pins the widest rung's algorithm; a stored table pins the shape's.
            // Both go through AlgoCheck, so a stale or foreign entry is refused here rather
            // than at launch.
            let stored = self.stored.get(&(m, n, k)).copied();
            let pinned = template.map(|t| t.algo).or(stored);
            if let Some(algo) = pinned {
                plan.algo = algo;
                let mut result = Heuristic::default();
                check(
                    (self.api.cublasLtMatmulAlgoCheck)(
                        self.handle as Handle,
                        plan.desc as Handle,
                        plan.w as Handle,
                        plan.a as Handle,
                        plan.c as Handle,
                        plan.c as Handle,
                        &plan.algo,
                        &mut result,
                    ),
                    "Lt rung algorithm",
                )?;
                let fits = result.state == 0 && result.workspace <= self.workspace.len as usize;
                if fits {
                    if template.is_none() {
                        tracing::info!(m, n, k, workspace = result.workspace, "cuBLASLt stored algorithm pinned");
                    }
                    return Ok(Arc::new(plan));
                }
                if template.is_some() {
                    return Err(RuntimeError::Rejected(
                        "cuBLASLt widest algorithm cannot serve narrower rung".into(),
                    ));
                }
                // A stored entry from another GPU or library version: fall through to the
                // heuristic + load-time timing, which is exactly what produced the table.
                tracing::warn!(m, n, k, "cuBLASLt stored algorithm rejected by AlgoCheck; re-selecting");
                plan.algo = Algo::default();
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

    /// `ld >= n` is the row pitch of the score matrix in ITS element type: `Scores` writes f32
    /// scores when `scores_f32` (bf16 x bf16 -> f32), `Values` always reads bf16 probabilities.
    /// The shape is only known at launch, so the algorithm is the first heuristic result: no
    /// load-time timing, no stored table.
    pub(crate) fn attention_plan(
        self: &Arc<Self>,
        kind: AttentionGemm,
        scores_f32: bool,
        m: u32,
        n: u32,
        ld: u32,
        hd: u32,
    ) -> Result<Plan> {
        self.be.bind()?;
        let mut plan = Plan {
            lt: self.clone(),
            shape: (m, n, hd),
            desc: 0,
            w: 0,
            a: 0,
            c: 0,
            algo: Algo::default(),
        };
        let (m, n, ld, hd) = (m as u64, n as u64, ld as i64, hd as u64);
        // CUDA header dtypes: BF16 = 14, FP32 = 0.
        let layouts = match kind {
            AttentionGemm::Scores => [
                (14, hd, n, hd as i64),
                (14, hd, m, hd as i64),
                (if scores_f32 { 0 } else { 14 }, n, m, ld),
            ],
            AttentionGemm::Values => [
                (14, hd, n, hd as i64),
                (14, n, m, ld),
                (14, hd, m, hd as i64),
            ],
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: same CUDA C ABI constants as `plan`; Plan drops what was created on error.
        unsafe {
            check(
                (self.api.cublasLtMatmulDescCreate)(&mut raw, 68, 0),
                "Lt attention descriptor",
            )?;
            plan.desc = raw as usize;
            if kind == AttentionGemm::Scores {
                let trans = 1i32;
                check(
                    (self.api.cublasLtMatmulDescSetAttribute)(
                        raw,
                        3,
                        &trans as *const _ as *const c_void,
                        size_of::<i32>(),
                    ),
                    "Lt attention transpose",
                )?;
            }
            for (dst, (dtype, rows, cols, pitch)) in
                [&mut plan.w, &mut plan.a, &mut plan.c].into_iter().zip(layouts)
            {
                raw = std::ptr::null_mut();
                check(
                    (self.api.cublasLtMatrixLayoutCreate)(&mut raw, dtype, rows, cols, pitch),
                    "Lt attention layout",
                )?;
                *dst = raw as usize;
            }
            let mut pref = std::ptr::null_mut();
            check(
                (self.api.cublasLtMatmulPreferenceCreate)(&mut pref),
                "Lt preference",
            )?;
            let bytes = self.workspace.len as usize;
            let mut results = [Heuristic::default(); 4];
            let mut count = 0;
            let status = check(
                (self.api.cublasLtMatmulPreferenceSetAttribute)(
                    pref,
                    1,
                    &bytes as *const _ as *const c_void,
                    size_of::<usize>(),
                ),
                "Lt workspace",
            )
            .and_then(|()| {
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
                    "Lt attention heuristic",
                )
            });
            (self.api.cublasLtMatmulPreferenceDestroy)(pref);
            status?;
            plan.algo = results
                .get(..count as usize)
                .unwrap_or(&[])
                .iter()
                .find(|r| r.state == 0 && r.workspace <= bytes)
                .ok_or_else(|| {
                    RuntimeError::Device("no supported BF16 cuBLASLt attention algorithm".into())
                })?
                .algo;
        }
        Ok(plan)
    }
}

/// The two GEMMs of dense attention over row-major BF16 operands with one KV head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AttentionGemm {
    /// `S[m][ld] = alpha * Q[m][hd] . K[n][hd]^T`; `matmul` operands `(K, Q, S)`.
    Scores,
    /// `O[m][hd] = P[m][ld] . V[n][hd]`; `matmul` operands `(V, P, O)`.
    Values,
}

/// Device-side operands of a grouped matmul `C_g[m_g, n] = A_g[m_g, k] * W_g[n, k]^T` (row-major).
/// Every array has `groups` entries and lives on the device for the plan's lifetime.
pub(crate) struct GroupedDims {
    pub groups: u32,
    pub n: u32,
    pub k: u32,
    /// `i32`: `m_g`, rewritten on the device before each launch.
    pub rows: u64,
    /// `i32` constants: every entry `n`, every entry `k`.
    pub n_array: u64,
    pub k_array: u64,
    /// Expected `m_g`; only steers the heuristic.
    pub average_rows: u32,
}

/// Device `i32[groups]` arrays of a grouped attention GEMM's per-group shapes (see
/// [`AttentionGemm`]): `hd` every entry the head dim, `m` score rows, `n` the 8-aligned KV
/// extent, `ld_s` / `ld_p` the score row pitch in S / P elements. Rewritten on the device
/// before each launch; a group with `m = 0` does nothing.
pub(crate) struct GroupedAttention {
    pub groups: u32,
    pub head_dim: u32,
    pub hd: u64,
    pub m: u64,
    pub n: u64,
    pub ld_s: u64,
    pub ld_p: u64,
    /// Expected shapes; only steer the heuristic.
    pub average_m: u32,
    pub average_n: u32,
}

/// One column-major grouped layout: `(dtype, rows, cols, ld)` device `i32` arrays.
type GroupedLayout = (i32, u64, u64, u64);

impl Lt {
    pub(crate) fn has_grouped(&self) -> bool {
        self.api.cublasLtGroupedMatrixLayoutCreate.is_some()
    }

    pub(crate) fn grouped_plan(self: &Arc<Self>, dims: &GroupedDims) -> Result<Arc<GroupedPlan>> {
        // Column-major views of the row-major operands: W is k x n, A is k x m_g, C is n x m_g.
        let mut plan = self.grouped_plan_raw(
            true,
            1.0,
            dims.groups,
            [
                (14, dims.k_array, dims.n_array, dims.k_array),
                (14, dims.k_array, dims.rows, dims.k_array),
                (14, dims.n_array, dims.rows, dims.n_array),
            ],
            [
                u64::from(dims.k),
                u64::from(dims.n),
                u64::from(dims.average_rows.max(1)),
            ],
        )?;
        plan.select(0);
        tracing::info!(
            groups = dims.groups,
            n = dims.n,
            k = dims.k,
            "cuBLASLt grouped algorithm selected"
        );
        Ok(Arc::new(plan))
    }

    /// The grouped form of [`Self::attention_plan`]: `alpha` scales the scores. The plan comes
    /// back unselected: every heuristic candidate is a [`GroupedPlan::candidates`] entry.
    pub(crate) fn grouped_attention_plan(
        self: &Arc<Self>,
        kind: AttentionGemm,
        scores_f32: bool,
        alpha: f32,
        dims: &GroupedAttention,
    ) -> Result<GroupedPlan> {
        let hd = u64::from(dims.head_dim);
        let (layouts, averages) = match kind {
            AttentionGemm::Scores => (
                [
                    (14, dims.hd, dims.n, dims.hd),
                    (14, dims.hd, dims.m, dims.hd),
                    (if scores_f32 { 0 } else { 14 }, dims.n, dims.m, dims.ld_s),
                ],
                [hd, u64::from(dims.average_n), u64::from(dims.average_m)],
            ),
            AttentionGemm::Values => (
                [
                    (14, dims.hd, dims.n, dims.hd),
                    (14, dims.n, dims.m, dims.ld_p),
                    (14, dims.hd, dims.m, dims.hd),
                ],
                [u64::from(dims.average_n), hd, u64::from(dims.average_m)],
            ),
        };
        self.grouped_plan_raw(kind == AttentionGemm::Scores, alpha, dims.groups, layouts, averages)
    }

    /// `layouts` = `[W, A, C]`; `averages` = `[reduction dim, D rows, D cols]` for the heuristic.
    /// The plan's `candidates` are every runnable heuristic result, best-first by the heuristic.
    fn grouped_plan_raw(
        self: &Arc<Self>,
        transpose_w: bool,
        alpha: f32,
        groups: u32,
        layouts: [GroupedLayout; 3],
        averages: [u64; 3],
    ) -> Result<GroupedPlan> {
        let create = self.api.cublasLtGroupedMatrixLayoutCreate.ok_or_else(|| {
            RuntimeError::Rejected(
                "cuBLASLt has no grouped matmul (Hopper needs cuBLAS 13.4 or newer)".into(),
            )
        })?;
        self.be.bind()?;
        let scalars = self.be.alloc(0, 8)?;
        let host: [f32; 2] = [alpha, 0.0];
        self.be.upload(&scalars, 0, bytemuck::cast_slice(&host))?;
        let mut plan = GroupedPlan {
            lt: self.clone(),
            desc: 0,
            w: 0,
            a: 0,
            c: 0,
            algo: Algo::default(),
            candidates: Vec::new(),
            scalars,
        };
        let mut raw = std::ptr::null_mut();
        // CUDA headers: BF16=14, FP32=0, COMPUTE_32F=68, scale FP32=0, POINTER_MODE attribute=2
        // (DEVICE=1), TRANSA attribute=3 (OP_T=1), preference MAX_WORKSPACE=1,
        // GROUPED_AVERAGE_REDUCTION_DIM=13, GROUPED_DESC_D_AVERAGE_ROWS=14 / _COLS=15.
        // SAFETY: exact CUDA 13 C ABI; GroupedPlan drops any descriptors created before an error.
        unsafe {
            check(
                (self.api.cublasLtMatmulDescCreate)(&mut raw, 68, 0),
                "Lt grouped descriptor",
            )?;
            plan.desc = raw as usize;
            let attributes: &[(i32, i32)] = if transpose_w {
                &[(3, 1), (2, 1)]
            } else {
                &[(2, 1)]
            };
            for &(attribute, value) in attributes {
                check(
                    (self.api.cublasLtMatmulDescSetAttribute)(
                        raw,
                        attribute,
                        &value as *const _ as *const c_void,
                        size_of::<i32>(),
                    ),
                    "Lt grouped descriptor attribute",
                )?;
            }
            for (dst, (dtype, rows, cols, ld)) in
                [&mut plan.w, &mut plan.a, &mut plan.c].into_iter().zip(layouts)
            {
                raw = std::ptr::null_mut();
                check(
                    create(
                        &mut raw,
                        dtype,
                        groups as i32,
                        rows as *const c_void,
                        cols as *const c_void,
                        ld as *const c_void,
                    ),
                    "Lt grouped layout",
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
                // The 13.4 header declares these `uint32_t`; the library's storage is 8 bytes
                // and a 4-byte write is rejected with CUBLAS_STATUS_INVALID_VALUE.
                for (attribute, value) in [13, 14, 15].into_iter().zip(averages) {
                    check(
                        (self.api.cublasLtMatmulPreferenceSetAttribute)(
                            pref,
                            attribute,
                            &value.max(1) as *const _ as *const c_void,
                            size_of::<u64>(),
                        ),
                        "Lt grouped average shape",
                    )?;
                }
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
                    "Lt grouped heuristic",
                )?;
                plan.candidates = results
                    .get(..count as usize)
                    .unwrap_or(&[])
                    .iter()
                    .filter(|r| r.state == 0 && r.workspace <= bytes)
                    .map(|r| r.algo)
                    .collect();
                if plan.candidates.is_empty() {
                    return Err(RuntimeError::Device(
                        "no supported BF16 cuBLASLt grouped algorithm".into(),
                    ));
                }
                tracing::debug!(
                    groups,
                    ?averages,
                    candidates = plan.candidates.len(),
                    "cuBLASLt grouped algorithms found"
                );
                Ok(())
            })();
            (self.api.cublasLtMatmulPreferenceDestroy)(pref);
            result?;
        }
        Ok(plan)
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
    shape: (u32, u32, u32),
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
        let matmul_ms = best / (repeats * 2) as f32;
        tracing::info!(
            m,
            n,
            k,
            index,
            candidates = candidates.len(),
            workspace = candidates[index].workspace,
            cold_bytes = copies.len,
            matmul_ms,
            "cuBLASLt load-time algorithm selected"
        );
        self.lt
            .record(m, n, k, &self.algo, candidates[index].workspace, matmul_ms * 1000.0);
        Ok(())
    }

    pub(crate) fn run(&self, a: u64, w: u64, c: u64, stream: &CudaStream) -> Result<()> {
        self.matmul(1.0, w, a, c, stream)
    }

    /// `c = alpha * op(w) * a` in the plan's layouts (cuBLAS operand order).
    pub(crate) fn matmul(
        &self,
        alpha: f32,
        w: u64,
        a: u64,
        c: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.matmul_beta(alpha, 0.0, w, a, c, stream)
    }

    /// `beta = 1` accumulates into `c` (C and D alias, same layout).
    pub(crate) fn matmul_beta(
        &self,
        alpha: f32,
        beta: f32,
        w: u64,
        a: u64,
        c: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.lt.be.bind()?;
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

pub(crate) struct GroupedPlan {
    lt: Arc<Lt>,
    desc: usize,
    w: usize,
    a: usize,
    c: usize,
    algo: Algo,
    /// Heuristic results still to choose from (`select` empties it into `algo`).
    candidates: Vec<Algo>,
    /// Device `[alpha, beta]`: a grouped matmul runs in device pointer mode.
    scalars: DeviceMem,
}

impl GroupedPlan {
    pub(crate) fn candidates(&self) -> usize {
        self.candidates.len()
    }

    pub(crate) fn select(&mut self, index: usize) {
        self.algo = self.candidates[index];
        self.candidates.clear();
    }

    /// `a`, `w`, `c` are device arrays of `groups` matrix pointers.
    pub(crate) fn run(&self, a: u64, w: u64, c: u64, stream: &CudaStream) -> Result<()> {
        self.run_algo(&self.algo, a, w, c, stream)
    }

    pub(crate) fn run_candidate(
        &self,
        index: usize,
        a: u64,
        w: u64,
        c: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_algo(&self.candidates[index], a, w, c, stream)
    }

    fn run_algo(&self, algo: &Algo, a: u64, w: u64, c: u64, stream: &CudaStream) -> Result<()> {
        self.lt.be.bind()?;
        // SAFETY: the route validated the pointer arrays' extents at load; the group shapes
        // are read on the device. All work is serialized on the engine stream.
        unsafe {
            check(
                (self.lt.api.cublasLtMatmul)(
                    self.lt.handle as Handle,
                    self.desc as Handle,
                    self.scalars.base as *const c_void,
                    w as *const c_void,
                    self.w as Handle,
                    a as *const c_void,
                    self.a as Handle,
                    (self.scalars.base + 4) as *const c_void,
                    c as *const c_void,
                    self.c as Handle,
                    c as *mut c_void,
                    self.c as Handle,
                    algo,
                    self.lt.workspace.base as *mut c_void,
                    self.lt.workspace.len as usize,
                    stream.raw as Handle,
                ),
                "cublasLtMatmul (grouped)",
            )
        }
    }
}

impl Drop for GroupedPlan {
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
