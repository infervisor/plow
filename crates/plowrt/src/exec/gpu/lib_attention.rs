//! Library flash prefill attention for head-dim-512 global layers (Gemma-4): FlashAttention-4's
//! SM90 forward, exported AOT into `libplow_fa4_hd512.so` (scripts/build_sm90a_fa4_hd512.py),
//! reading 64-row pages through a page table. The packet declares the object
//! (`LIBRARY_PREFILL_ATTENTION`) and, per program, the segments it serves; their native role is
//! the fallback for a launch outside the library's envelope (`begin_launch` decides before any
//! work is enqueued).
//!
//! FP8 KV (one f32 scale per (position, KV head) row) has no library form, so per routed site the
//! library's own gather (`plow_fa4_kv_gather`) first copies every request's rows `[0, kv_len)`
//! into the pool (`[kv head][page][64][512]`, the request's pages consecutive). The gather never
//! reads a slot row at or past kv_len (live KV may leave it unmapped). A BF16 cache is read in
//! place; `plow_fa4_kv_zero_tail` zeroes rows `[kv_len, round_up(kv_len, 64))` first. Those rows
//! are private to the request: they share the 64-row page, and so the live-KV block (a power of
//! two >= the 2 MiB granularity, i.e. a multiple of 64 rows), with row kv_len - 1, which this
//! launch's KV append just wrote; they are the slot's next append rows.
//!
//! One table upload per launch (`begin_launch`): cu_seqlens_q, kv lengths, the page table and the
//! gather list; every routed site of the launch reuses it.

use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use std::ffi::c_void;

const PAGE: u32 = 64;
const HEAD_DIM: u32 = 512;
const HEADS: u32 = 16;
/// Pool rows per KV head: one launch's summed kv lengths, page-rounded. A launch past it runs the
/// native role.
pub(super) const POOL_ROWS: u32 = 65536;
const POOL_PAGES: u32 = POOL_ROWS / PAGE;
/// Requests one launch may carry (the device table is sized for it).
const MAX_REQUESTS: usize = 256;
const STAGING: usize = 4;

type Create = unsafe extern "C" fn(i32, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Abi = unsafe extern "C" fn() -> u32;
type Dtype = unsafe extern "C" fn() -> u32;
type Gather = unsafe extern "C" fn(*const DequantArgs, *mut c_void) -> i32;
type ZeroTail = unsafe extern "C" fn(*const ZeroTailArgs, *mut c_void) -> i32;
type Convert = unsafe extern "C" fn(*mut c_void, u64, u32, *mut c_void) -> i32;
type Run = unsafe extern "C" fn(
    *mut c_void, // handle
    u32,         // heads
    u32,         // kv heads
    u64,         // q
    u64,         // k pages
    u64,         // v pages
    u64,         // out
    i32,         // total q rows
    i32,         // requests
    u64,         // cu_seqlens_q
    u64,         // kv lengths
    u64,         // page table
    i32,         // page table width
    i32,         // pages
    i64,         // KV head stride (elements)
    f32,         // softmax scale
    i32,         // KV splits (1 = whole-KV entry)
    u64,         // split workspace
    u64,         // workspace bytes
    *mut c_void, // stream
) -> i32;
/// Split-KV partials: f32 O + LSE per (split, row, head).
const SPLIT_ROW_BYTES: u64 = 16 * (512 * 4 + 4);
const WORKSPACE_BYTES: u64 = 128 << 20;
const SMS: u32 = 132;

/// One routed `FlashPrefill(Fp8)` site. Addresses are the slot-0 tensor bases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Site {
    pub(super) instruction: usize,
    q: u64,
    k: u64,
    v: u64,
    /// The per-row K / V scale tensors of an FP8 cache; `None` = BF16 cache, read in place.
    scales: Option<(u64, u64)>,
    output: u64,
    kv_heads: u32,
    kv_stride: u32,
    scale: f32,
}

impl Site {
    pub(super) fn kv_heads(&self) -> u32 {
        self.kv_heads
    }
    /// `(kv heads, rows per slot)` of a BF16 site (the direct, adapter-free path).
    pub(super) fn direct_geometry(&self) -> Option<(u32, u32)> {
        self.scales
            .is_none()
            .then_some((self.kv_heads, self.kv_stride))
    }
}

/// Segments whose single instruction is a causal global hd512 `FlashPrefill(Fp8)` with 16 query
/// heads over 1 or 2 KV heads on a head-major linear cache. Anything else keeps its launch.
pub(super) fn sites(
    program: &DevProg,
    tensors: &[DevTensor],
    devp: &[DeviceMem],
    batch: usize,
) -> Vec<Option<Site>> {
    let count = program.gq_seg_ofs.len().saturating_sub(1);
    let mut out = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let Some(entries) = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
        else {
            continue;
        };
        let Some(instruction) = entries.first().map(|e| e.inst as usize) else {
            continue;
        };
        let Some(op) = program.insts.get(instruction) else {
            continue;
        };
        let fp8 = op.op == DevOp::FlashPrefillFp8 as u16;
        let bf16 = op.op == DevOp::FlashPrefill as u16;
        let [rows, _, heads, kv_heads, _, window, head_dim, nsplit] = op.i;
        let kv_stride = op.fj[1];
        if !(fp8 || bf16)
            || heads != HEADS
            || !(kv_heads == 1 || kv_heads == 2)
            || head_dim != HEAD_DIM
            || window != 0
            || nsplit != 1
            || kv_stride == 0
            || kv_stride % PAGE != 0
            || op.fj[2] != u32::MAX
            || op.t[2..6].contains(&TENSOR_NONE16)
            || (fp8 && op.t[6..8].contains(&TENSOR_NONE16))
            || entries.iter().any(|e| e.inst as usize != instruction)
            || program
                .stream
                .iter()
                .any(|e| (e.inst as usize == instruction) != (e.seg as usize == segment))
        {
            continue;
        }
        let element = if fp8 { 1 } else { 2 };
        let slot_bytes = u64::from(kv_heads) * u64::from(kv_stride) * u64::from(HEAD_DIM) * element;
        let row_bytes = u64::from(HEADS * HEAD_DIM) * 2;
        let fits = |handle: u16, bytes: u64| {
            tensors
                .get(handle as usize)
                .is_some_and(|t| t.bytes >= bytes)
                && devp.get(handle as usize).is_some()
        };
        let scale_bytes = slot_bytes / u64::from(HEAD_DIM) * 4;
        if !fits(op.t[2], u64::from(rows) * row_bytes)
            || !fits(op.t[5], u64::from(rows) * row_bytes)
            || !fits(op.t[3], batch as u64 * slot_bytes)
            || !fits(op.t[4], batch as u64 * slot_bytes)
            || (fp8 && !fits(op.t[6], batch as u64 * scale_bytes))
            || (fp8 && !fits(op.t[7], batch as u64 * scale_bytes))
        {
            continue;
        }
        let p = |h: u16| devp[h as usize].base;
        out[segment] = Some(Site {
            instruction,
            q: p(op.t[2]),
            k: p(op.t[3]),
            v: p(op.t[4]),
            scales: fp8.then(|| (p(op.t[6]), p(op.t[7]))),
            output: p(op.t[5]),
            kv_heads,
            kv_stride,
            scale: f32::from_bits(op.fj[0]),
        });
    }
    out
}

/// `plow_kv_dequant_pages`' argument block (runtime/nvidia/kv_dequant_sm90.cu).
#[repr(C)]
#[derive(Clone, Copy)]
struct DequantArgs {
    k8: u64,
    ks: u64,
    v8: u64,
    vs: u64,
    ko: u64,
    vo: u64,
    pages: u64,
    n_pages: u32,
    kv_heads: u32,
    kv_stride: u32,
    pool_pages: u32,
    f16: u32,
}
const _: () = assert!(std::mem::size_of::<DequantArgs>() == 80);

struct Staging {
    host: PinnedHost,
    done: CudaEvent,
    pending: bool,
}

/// The current launch's table, uploaded by `begin_launch`.
#[derive(Clone, Copy)]
struct Launch {
    rows: u32,
    requests: u32,
    width: u32,
    pages: u32,
    /// Longest query and KV extents of the launch's requests (the split heuristic).
    max_q: u32,
    max_kv: u32,
    /// Output rows past the last request: zeroed once per site.
    tail: (u64, u64),
    /// First request row (riders, if any, precede it).
    head: u64,
}

impl Launch {
    /// vLLM's FA4 split rule (`num_splits_heuristic`): fill the SMs with (64-row query tile,
    /// head) blocks, never more splits than 64-row KV blocks, none at <= 4 KV blocks; capped by
    /// the workspace. The export packs GQA only at 2 KV heads; at 1 every query head is its own
    /// block row.
    fn splits(&self, kv_heads: u32, workspace: u64) -> u32 {
        let n_blocks = self.max_kv.div_ceil(PAGE);
        let m_blocks = if kv_heads == 2 {
            self.requests * kv_heads * (self.max_q * (HEADS / kv_heads)).div_ceil(64)
        } else {
            self.requests * HEADS * self.max_q.div_ceil(64)
        };
        if n_blocks <= 4 || m_blocks == 0 {
            return 1;
        }
        let mut splits = (SMS / m_blocks).min(128).min(n_blocks).max(1);
        while splits > 1 && u64::from(splits) * u64::from(self.rows) * SPLIT_ROW_BYTES > workspace {
            splits -= 1;
        }
        splits
    }
}

pub(super) struct LibAttention {
    be: Arc<CudaBackend>,
    /// `plow_fa4_create`'s handle, destroyed on unload or drop (before the library unmaps).
    handle: Option<usize>,
    _library: libloading::Library,
    destroy: Destroy,
    run: Run,
    gather_fn: Gather,
    zero_tail: ZeroTail,
    convert_fn: Convert,
    /// The library's element type is fp16 (`plow_fa4_dtype` 1): the pool is fp16 and Q / O are
    /// converted in place around each call.
    f16: bool,
    /// FP8 sites' gather pool (empty in the direct mode).
    pool_k: DeviceMem,
    pool_v: DeviceMem,
    /// BF16 cache read in place: pages are the cache's own 64-row runs, `slot_pages` per slot
    /// (`kv heads * rows per slot / 64`, head stride `rows per slot` rows), `total_pages` in all.
    direct: Option<Direct>,
    /// Device table: cu_q [MAX_REQUESTS + 1] | kv_len [MAX_REQUESTS] | gather [POOL_PAGES][3] |
    /// page table [MAX_REQUESTS][width].
    table: DeviceMem,
    /// Split-KV partials (`plow_fa4_hd512_workspace`).
    workspace: DeviceMem,
    staging: Vec<Staging>,
    staging_next: usize,
    launch: Option<Launch>,
    max_kv_heads: u32,
    fallbacks: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Direct {
    pub(super) kv_heads: u32,
    pub(super) kv_stride: u32,
    pub(super) slots: u32,
}

/// `plow_kv_zero_tail`'s argument block (runtime/nvidia/kv_dequant_sm90.cu).
#[repr(C)]
#[derive(Clone, Copy)]
struct ZeroTailArgs {
    k: u64,
    v: u64,
    list: u64,
    n: u32,
    kv_heads: u32,
    kv_stride: u32,
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<ZeroTailArgs>() == 40);

const CU_OFF: u64 = 0;
const LEN_OFF: u64 = (MAX_REQUESTS as u64 + 1) * 4;
const GATHER_OFF: u64 = LEN_OFF + MAX_REQUESTS as u64 * 4;
const PT_OFF: u64 = GATHER_OFF + POOL_PAGES as u64 * 12;
/// The page table holds at most every pool page once per request row width.
const TABLE_BYTES: u64 = PT_OFF + MAX_REQUESTS as u64 * POOL_PAGES as u64 * 4;

fn check(rc: i32, what: &str) -> Result<()> {
    match rc {
        0 => Ok(()),
        -1001 => Err(RuntimeError::Device(format!(
            "{what}: the library already has a live handle (one engine per loaded library)"
        ))),
        rc => Err(RuntimeError::Device(format!("{what}: status {rc}"))),
    }
}

impl LibAttention {
    /// The packet's declared library object (`assets_dir`-relative), hash- and ABI-checked.
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        object: &SegmentObject,
        max_kv_heads: u32,
        direct: Option<Direct>,
    ) -> Result<Self> {
        let library = assets_dir.join(&object.file);
        let image = std::fs::read(&library)
            .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", library.display())))?;
        if object.sha256.as_deref()
            != Some(plow_asset::decode_objects::image_sha256(&image).as_str())
        {
            return Err(RuntimeError::Rejected(format!(
                "{}: library attention object hash mismatch",
                library.display()
            )));
        }
        let declared_f16 = match object.abi.as_str() {
            plow_asset::segment_roles::LIBRARY_FA4_HD512_BF16_ABI => false,
            plow_asset::segment_roles::LIBRARY_FA4_HD512_F16_ABI => true,
            abi => {
                return Err(RuntimeError::Rejected(format!(
                    "library attention ABI {abi} unsupported"
                )))
            }
        };
        be.bind()?;
        // SAFETY: the hash-pinned library implements the plow_fa4 C ABI (3, checked below).
        let lib = unsafe { libloading::Library::new(&library) }
            .map_err(|e| RuntimeError::Device(format!("load {}: {e}", library.display())))?;
        let sym = |name: &[u8]| {
            RuntimeError::Device(format!(
                "{} lacks {}",
                library.display(),
                String::from_utf8_lossy(&name[..name.len() - 1])
            ))
        };
        // SAFETY: symbol types match runtime/nvidia/fa4_hd512.cpp and kv_dequant_sm90.cu.
        let (abi, dtype, create, destroy, run, gather_fn, zero_tail, convert_fn) = unsafe {
            (
                *lib.get::<Abi>(b"plow_fa4_abi\0")
                    .map_err(|_| sym(b"plow_fa4_abi\0"))?,
                *lib.get::<Dtype>(b"plow_fa4_dtype\0")
                    .map_err(|_| sym(b"plow_fa4_dtype\0"))?,
                *lib.get::<Create>(b"plow_fa4_create\0")
                    .map_err(|_| sym(b"plow_fa4_create\0"))?,
                *lib.get::<Destroy>(b"plow_fa4_destroy\0")
                    .map_err(|_| sym(b"plow_fa4_destroy\0"))?,
                *lib.get::<Run>(b"plow_fa4_hd512_run\0")
                    .map_err(|_| sym(b"plow_fa4_hd512_run\0"))?,
                *lib.get::<Gather>(b"plow_fa4_kv_gather\0")
                    .map_err(|_| sym(b"plow_fa4_kv_gather\0"))?,
                *lib.get::<ZeroTail>(b"plow_fa4_kv_zero_tail\0")
                    .map_err(|_| sym(b"plow_fa4_kv_zero_tail\0"))?,
                *lib.get::<Convert>(b"plow_fa4_convert\0")
                    .map_err(|_| sym(b"plow_fa4_convert\0"))?,
            )
        };
        // SAFETY: no arguments.
        let (abi, dtype) = unsafe { (abi(), dtype()) };
        if abi != 3 || dtype != u32::from(declared_f16) {
            return Err(RuntimeError::Rejected(format!(
                "{}: plow_fa4 ABI {abi} dtype {dtype}; the packet declares ABI 3 {}",
                library.display(),
                object.abi
            )));
        }
        let f16 = declared_f16;
        if let Some(d) = direct {
            if f16 {
                return Err(RuntimeError::Rejected(format!(
                    "{}: a BF16 KV cache needs the bf16 library",
                    library.display()
                )));
            }
            if d.kv_stride % PAGE != 0
                || u64::from(d.slots) * u64::from(d.kv_heads) * u64::from(d.kv_stride)
                    / u64::from(PAGE)
                    > i32::MAX as u64
            {
                return Err(RuntimeError::Rejected(
                    "library attention: cache geometry".into(),
                ));
            }
        }
        let pool_bytes = if direct.is_some() {
            16
        } else {
            u64::from(max_kv_heads) * u64::from(POOL_ROWS) * u64::from(HEAD_DIM) * 2
        };
        let pool_k = be.alloc(0, pool_bytes)?;
        let pool_v = be.alloc(0, pool_bytes)?;
        let table = be.alloc(0, TABLE_BYTES)?;
        let workspace = be.alloc(0, WORKSPACE_BYTES)?;
        let mut staging = Vec::with_capacity(STAGING);
        for _ in 0..STAGING {
            staging.push(Staging {
                host: be.host_alloc_pinned(TABLE_BYTES as usize)?,
                done: be.event_create(false)?,
                pending: false,
            });
        }
        let mut handle = std::ptr::null_mut();
        // SAFETY: out-pointer valid for the call.
        check(
            unsafe { create(be.device_ordinal as i32, &mut handle) },
            "plow_fa4_create",
        )?;
        if handle.is_null() {
            return Err(RuntimeError::Device("plow_fa4_create returned null".into()));
        }
        tracing::info!(
            library = %library.display(),
            pool_mib = (2 * pool_bytes) >> 20,
            pool = if direct.is_some() { "none (BF16 cache in place)" } else if f16 { "fp16" } else { "bf16" },
            "library hd512 prefill attention (FA4) loaded"
        );
        Ok(Self {
            be: Arc::clone(be),
            handle: Some(handle as usize),
            _library: lib,
            destroy,
            run,
            gather_fn,
            zero_tail,
            convert_fn,
            f16,
            pool_k,
            pool_v,
            direct,
            table,
            workspace,
            staging,
            staging_next: 0,
            launch: None,
            max_kv_heads,
            fallbacks: 0,
        })
    }

    pub(super) fn unload(mut self) -> Result<()> {
        let rc = self.destroy_handle();
        check(rc, "plow_fa4_destroy")
    }

    fn destroy_handle(&mut self) -> i32 {
        // SAFETY: the handle came from plow_fa4_create and is taken, so destroyed once.
        self.handle
            .take()
            .map_or(0, |h| unsafe { (self.destroy)(h as *mut c_void) })
    }

    /// Builds and uploads the launch's table. `false` = the launch is outside the envelope
    /// (requests, pool pages, slots); its sites run their declared native role. Nothing is
    /// enqueued before the decision.
    pub(super) fn begin_launch(
        &mut self,
        requests: &[attention_gemm::Request],
        rows: u32,
        stream: &CudaStream,
    ) -> Result<bool> {
        let routed = self.plan(requests, rows, stream)?;
        if !routed {
            self.fallbacks += 1;
            if self.fallbacks.is_power_of_two() {
                tracing::warn!(
                    fallbacks = self.fallbacks,
                    requests = requests.len(),
                    rows,
                    "library attention: launch outside the library envelope; native role runs"
                );
            }
        }
        Ok(routed)
    }

    fn plan(
        &mut self,
        requests: &[attention_gemm::Request],
        rows: u32,
        stream: &CudaStream,
    ) -> Result<bool> {
        self.launch = None;
        if requests.is_empty() || requests.len() > MAX_REQUESTS {
            return Ok(false);
        }
        let mut pages = 0u32;
        let mut width = 0u32;
        for &[q0, qlen, _, kvlen] in requests {
            if qlen == 0 || kvlen < qlen || q0.checked_add(qlen).is_none_or(|e| e > rows) {
                return Ok(false);
            }
            let n = kvlen.div_ceil(PAGE);
            width = width.max(n);
            pages += n;
        }
        if (self.direct.is_none() && pages > POOL_PAGES)
            || u64::from(width) * requests.len() as u64
                > u64::from(POOL_PAGES) * MAX_REQUESTS as u64
        {
            return Ok(false);
        }
        if let Some(d) = self.direct {
            if requests
                .iter()
                .any(|r| r[2] >= d.slots || r[3] > d.kv_stride)
            {
                return Ok(false);
            }
        }
        // Requests are contiguous query rows (riders, if any, precede them).
        let mut cu = vec![requests[0][0] as i32];
        for &[q0, qlen, ..] in requests {
            if *cu.last().expect("seeded") != q0 as i32 {
                return Ok(false);
            }
            cu.push((q0 + qlen) as i32);
        }
        let mut gather = Vec::with_capacity(pages as usize * 3);
        let mut table = vec![0i32; width as usize * requests.len()];
        for (r, &[_, _, slot_id, kvlen]) in requests.iter().enumerate() {
            if let Some(d) = self.direct {
                // The slot's own pages; the gather list carries (slot, kvlen) for the tail zeroing.
                let first = slot_id * (d.kv_heads * d.kv_stride / PAGE);
                for j in 0..kvlen.div_ceil(PAGE) {
                    table[r * width as usize + j as usize] = (first + j) as i32;
                }
                gather.extend([slot_id as i32, kvlen as i32, 0]);
                continue;
            }
            for j in 0..kvlen.div_ceil(PAGE) {
                table[r * width as usize + j as usize] = (gather.len() / 3) as i32;
                gather.extend([
                    slot_id as i32,
                    (j * PAGE) as i32,
                    (kvlen - j * PAGE).min(PAGE) as i32,
                ]);
            }
        }
        let lens: Vec<i32> = requests.iter().map(|r| r[3] as i32).collect();
        let pt_bytes = table.len() as u64 * 4;
        let slot = self.staging_next;
        self.staging_next = (slot + 1) % STAGING;
        let staging = &mut self.staging[slot];
        if staging.pending {
            self.be.event_synchronize(&staging.done)?;
            staging.pending = false;
        }
        let host = staging.host.as_mut_slice();
        for (off, values) in [
            (CU_OFF, &cu),
            (LEN_OFF, &lens),
            (GATHER_OFF, &gather),
            (PT_OFF, &table),
        ] {
            for (i, v) in values.iter().enumerate() {
                let at = off as usize + 4 * i;
                host[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let regions = [
            (CU_OFF, CU_OFF + cu.len() as u64 * 4),
            (LEN_OFF, LEN_OFF + lens.len() as u64 * 4),
            (GATHER_OFF, GATHER_OFF + gather.len() as u64 * 4),
            (PT_OFF, PT_OFF + pt_bytes),
        ];
        let upload = || -> Result<()> {
            for (start, end) in regions {
                // SAFETY: the pinned staging stays untouched until `done` (recorded below) fires.
                unsafe {
                    self.be.memcpy_htod_async(
                        self.table.base + start,
                        &staging.host.as_slice()[start as usize..end as usize],
                        stream,
                    )?;
                }
            }
            self.be.event_record(&staging.done, stream)
        };
        if let Err(e) = upload() {
            // A copy may be in flight: drain before the slot can be refilled.
            let _ = self.be.stream_synchronize(stream);
            return Err(e);
        }
        staging.pending = true;
        let last = requests.last().expect("non-empty");
        self.launch = Some(Launch {
            rows,
            requests: requests.len() as u32,
            width,
            pages,
            max_q: requests.iter().map(|r| r[1]).max().unwrap_or(1),
            max_kv: requests.iter().map(|r| r[3]).max().unwrap_or(1),
            tail: (u64::from(last[0] + last[1]), u64::from(rows)),
            head: u64::from(requests[0][0]),
        });
        Ok(true)
    }

    pub(super) fn run_site(&self, site: &Site, stream: &CudaStream) -> Result<()> {
        let l = self
            .launch
            .ok_or_else(|| RuntimeError::Rejected("library attention without a launch".into()))?;
        if site.kv_heads > self.max_kv_heads {
            return Err(RuntimeError::Rejected(
                "library attention: KV heads exceed the pool".into(),
            ));
        }
        let handle = self
            .handle
            .ok_or_else(|| RuntimeError::Rejected("library attention unloaded".into()))?;
        let (k_pages, v_pages, num_pages, head_stride) = match (self.direct, site.scales) {
            (Some(d), None) => {
                // Rows past each kvlen up to its page end: zero, since the library loads whole
                // pages and 0 x NaN would poison P.V (ownership: module doc).
                let args = ZeroTailArgs {
                    k: site.k,
                    v: site.v,
                    list: self.table.base + GATHER_OFF,
                    n: l.requests,
                    kv_heads: site.kv_heads,
                    kv_stride: site.kv_stride,
                    _pad: 0,
                };
                // SAFETY: args lives across the call; the launch copies it.
                check(
                    unsafe { (self.zero_tail)(&args, stream.raw() as *mut c_void) },
                    "plow_fa4_kv_zero_tail",
                )?;
                (
                    site.k,
                    site.v,
                    d.slots * d.kv_heads * d.kv_stride / PAGE,
                    i64::from(d.kv_stride) * i64::from(HEAD_DIM),
                )
            }
            (None, Some(scales)) => {
                self.gather(site, scales, &l, stream)?;
                (
                    self.pool_k.base,
                    self.pool_v.base,
                    POOL_PAGES,
                    i64::from(POOL_PAGES) * i64::from(PAGE * HEAD_DIM),
                )
            }
            _ => {
                return Err(RuntimeError::Rejected(
                    "library attention: site KV type differs from the engine's".into(),
                ))
            }
        };
        let row_bytes = u64::from(HEADS * HEAD_DIM) * 2;
        let (t0, t1) = l.tail;
        if t1 > t0 {
            self.be.memset_d8_async(
                site.output + t0 * row_bytes,
                0,
                ((t1 - t0) * row_bytes) as usize,
                stream,
            )?;
        }
        let splits = l.splits(site.kv_heads, self.workspace.len);
        let rows_q = (site.q + l.head * row_bytes, (t0 - l.head) * row_bytes);
        let rows_o = (site.output + l.head * row_bytes, (t0 - l.head) * row_bytes);
        if self.f16 {
            self.convert(rows_q, true, stream)?;
        }
        // SAFETY: every pointer is a live device buffer sized above; the stream is ours.
        check(
            unsafe {
                (self.run)(
                    handle as *mut c_void,
                    HEADS,
                    site.kv_heads,
                    site.q,
                    k_pages,
                    v_pages,
                    site.output,
                    l.rows as i32,
                    l.requests as i32,
                    self.table.base + CU_OFF,
                    self.table.base + LEN_OFF,
                    self.table.base + PT_OFF,
                    l.width as i32,
                    num_pages as i32,
                    head_stride,
                    site.scale,
                    splits as i32,
                    self.workspace.base,
                    self.workspace.len,
                    stream.raw() as *mut c_void,
                )
            },
            "plow_fa4_hd512_run",
        )?;
        if self.f16 {
            self.convert(rows_o, false, stream)?;
            self.convert(rows_q, false, stream)?;
        }
        Ok(())
    }

    /// FP8 KV rows `[0, kvlen)` of every request into the pool (`plow_fa4_kv_gather`).
    fn gather(
        &self,
        site: &Site,
        (ks, vs): (u64, u64),
        l: &Launch,
        stream: &CudaStream,
    ) -> Result<()> {
        let args = DequantArgs {
            k8: site.k,
            ks,
            v8: site.v,
            vs,
            ko: self.pool_k.base,
            vo: self.pool_v.base,
            pages: self.table.base + GATHER_OFF,
            n_pages: l.pages,
            kv_heads: site.kv_heads,
            kv_stride: site.kv_stride,
            pool_pages: POOL_PAGES,
            f16: u32::from(self.f16),
        };
        // SAFETY: args lives across the call; the launch copies it.
        check(
            unsafe { (self.gather_fn)(&args, stream.raw() as *mut c_void) },
            "plow_fa4_kv_gather",
        )
    }

    /// In-place bf16 <-> fp16 of `(address, bytes)` (16 B multiple).
    fn convert(
        &self,
        (address, bytes): (u64, u64),
        to_f16: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        // SAFETY: a live device range of the routed site; the stream is ours.
        check(
            unsafe {
                (self.convert_fn)(
                    address as *mut c_void,
                    bytes / 16,
                    u32::from(to_f16),
                    stream.raw() as *mut c_void,
                )
            },
            "plow_fa4_convert",
        )
    }
}

impl Drop for LibAttention {
    fn drop(&mut self) {
        let _ = self.destroy_handle();
    }
}
