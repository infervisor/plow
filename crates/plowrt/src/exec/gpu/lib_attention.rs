//! Library flash prefill attention for causal global layers: FlashAttention-4's SM90 forward,
//! exported AOT as cubins into a packet object (`plow_fa4_hd512.aot`: hd 512 over a head-major
//! cache; `plow_fa4_hd256.aot`: hd 256 over a row-interleaved cache; scripts/fa4_aot/build.py)
//! and launched by plowrt itself (`fa4_sm90.rs`), reading 64-row pages through a page table. The
//! object's ABI ([`LibraryAbi`]) names the geometry it computes. The packet declares the object
//! (`LIBRARY_PREFILL_ATTENTION`) and, per program, the segments it serves; their native role is
//! the fallback for a launch outside the library's envelope (`begin_launch` decides before any
//! work is enqueued).
//!
//! FP8 KV (one f32 scale per (position, KV head) row) has no library form, so per routed site the
//! object's adapter gather (`plow_kv_dequant_pages`) first copies every request's rows `[0, kv_len)`
//! into the pool (`[kv head][page][64][512]`, the request's pages consecutive). The gather never
//! reads a slot row at or past kv_len (live KV may leave it unmapped). A BF16 cache is read in
//! place. The head-major entries load whole pages (TMA), so `plow_kv_zero_tail` zeroes rows
//! `[kv_len, round_up(kv_len, 64))` first; the row-interleaved entries load row-predicated (no
//! row at or past kv_len) and need none. The zeroed rows
//! are private to the request: they share the 64-row page, and so the live-KV block (a power of
//! two >= the 2 MiB granularity, i.e. a multiple of 64 rows), with row kv_len - 1, which this
//! launch's KV append just wrote; they are the slot's next append rows.
//!
//! One table upload per launch (`begin_launch`): cu_seqlens_q, kv lengths, the page table and the
//! gather list; every routed site of the launch reuses it.

use super::fa4_sm90::{Call, Fa4, Geometry, ADAPTERS};
use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::cuda::aot::Image;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use plow_asset::segment_roles::{LibraryAbi, LibraryKvLayout};

/// Process-wide launch counts across engines: the shutdown path exits without unloading.
static LOADED: AtomicBool = AtomicBool::new(false);
static SERVED: AtomicU64 = AtomicU64::new(0);
static FALLBACK: AtomicU64 = AtomicU64::new(0);

/// (served, fallback) launches of every library route this process loaded; `None` = none.
pub(crate) fn process_launch_counts() -> Option<(u64, u64)> {
    LOADED.load(Ordering::Relaxed).then(|| {
        (
            SERVED.load(Ordering::Relaxed),
            FALLBACK.load(Ordering::Relaxed),
        )
    })
}

/// Cadence of the served / fallback count line (also logged at unload).
const STATS_EVERY: std::time::Duration = std::time::Duration::from_secs(60);
const PAGE: u32 = 64;
/// Pool rows per KV head: one launch's summed kv lengths, page-rounded. A launch past it runs the
/// native role.
pub(super) const POOL_ROWS: u32 = 65536;
const POOL_PAGES: u32 = POOL_ROWS / PAGE;
/// Requests one launch may carry (the device table is sized for it).
const MAX_REQUESTS: usize = 256;
const STAGING: usize = 4;

const WORKSPACE_BYTES: u64 = 128 << 20;
/// Split-KV upper bound (the exported combine's log_max_splits 7).
const MAX_SPLITS: u32 = 128;
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
    heads: u32,
    kv_heads: u32,
    kv_stride: u32,
    scale: f32,
}

impl Site {
    pub(super) fn kv_heads(&self) -> u32 {
        self.kv_heads
    }
    /// Rows per slot of the site's cache.
    pub(super) fn rows(&self) -> u32 {
        self.kv_stride
    }
    /// `(kv heads, rows per slot)` of a BF16 site (the direct, adapter-free path).
    pub(super) fn direct_geometry(&self) -> Option<(u32, u32)> {
        self.scales
            .is_none()
            .then_some((self.kv_heads, self.kv_stride))
    }
}

/// Segments whose single instruction is a causal global `FlashPrefill(Fp8)` of the library's
/// geometry on a linear cache (FP8 KV only through the hd512 head-major gather). Anything else
/// keeps its launch.
pub(super) fn sites(
    program: &DevProg,
    tensors: &[DevTensor],
    devp: &[DeviceMem],
    batch: usize,
    abi: &LibraryAbi,
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
        if !(bf16 || (fp8 && gathers(abi)))
            || heads != abi.heads
            || !abi.kv_heads.contains(&kv_heads)
            || head_dim != abi.head_dim
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
        let slot_bytes = u64::from(kv_heads) * u64::from(kv_stride) * u64::from(head_dim) * element;
        let row_bytes = u64::from(heads * head_dim) * 2;
        let fits = |handle: u16, bytes: u64| {
            tensors
                .get(handle as usize)
                .is_some_and(|t| t.bytes >= bytes)
                && devp.get(handle as usize).is_some()
        };
        let scale_bytes = slot_bytes / u64::from(head_dim) * 4;
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
            heads,
            kv_heads,
            kv_stride,
            scale: f32::from_bits(op.fj[0]),
        });
    }
    out
}

/// The adapter ABI `DequantArgs` / `ZeroTailArgs` and the `plow_cvt_rows` parameters mirror.
const ADAPTER_ABI: u32 = 2;

/// The `kv_adapters` cubin states the adapter ABI this host code was written against.
fn check_adapter_abi(cubin: &[u8]) -> std::result::Result<(), String> {
    match plow_asset::cubin::global_u32(cubin, "plow_kv_dequant_abi") {
        Some(ADAPTER_ABI) => Ok(()),
        Some(v) => Err(format!("KV adapter ABI {v}, plowrt speaks {ADAPTER_ABI}")),
        None => Err("KV adapter cubin states no plow_kv_dequant_abi".into()),
    }
}

/// FP8 KV reaches a library only through the hd512 head-major page gather.
fn gathers(abi: &LibraryAbi) -> bool {
    abi.head_dim == 512 && abi.layout == LibraryKvLayout::HeadMajor
}

/// The exported entries' tiling (scripts/export_fa4_hd*_sm90.py, vLLM's SM90 choice), for the
/// split rule: query tile, KV tile, and whether query heads of a KV head pack into one tile.
fn tiles(abi: &LibraryAbi, kv_heads: u32) -> (u32, u32, bool) {
    match abi.head_dim {
        512 => (64, 64, kv_heads == 2),
        _ => (128, 80, abi.heads > kv_heads),
    }
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
    /// vLLM's FA4 split rule (`num_splits_heuristic`): fill the SMs with (query tile, head)
    /// blocks, never more splits than KV blocks, none at <= 4 KV blocks; capped by the
    /// workspace. The hd512 export packs GQA only at 2 KV heads; at 1 every query head is its
    /// own block row.
    fn splits(&self, abi: &LibraryAbi, kv_heads: u32, workspace: u64) -> u32 {
        let (tile_m, tile_n, pack) = tiles(abi, kv_heads);
        let n_blocks = self.max_kv.div_ceil(tile_n);
        let m_blocks = if pack {
            self.requests * kv_heads * (self.max_q * (abi.heads / kv_heads)).div_ceil(tile_m)
        } else {
            self.requests * abi.heads * self.max_q.div_ceil(tile_m)
        };
        // Split-KV partials: f32 O + LSE per (split, row, head).
        let split_row_bytes = u64::from(abi.heads) * (u64::from(abi.head_dim) * 4 + 4);
        if n_blocks <= 4 || m_blocks == 0 {
            return 1;
        }
        let mut splits = (SMS / m_blocks).min(MAX_SPLITS).min(n_blocks).max(1);
        while splits > 1 && u64::from(splits) * u64::from(self.rows) * split_row_bytes > workspace {
            splits -= 1;
        }
        splits
    }
}

/// The object's KV adapter kernels (runtime/nvidia/kv_dequant_sm90.cu).
struct Adapters {
    module: Option<Module>,
    gather: KernelFn,
    zero_tail: KernelFn,
    convert: KernelFn,
}
/// The adapters' block size (kv_dequant_sm90.cu THREADS).
const ADAPTER_THREADS: u32 = 256;

pub(super) struct LibAttention {
    be: Arc<CudaBackend>,
    fa4: Fa4,
    adapters: Adapters,
    /// The declared geometry; `f16`: the pool is fp16 and Q / O are converted in place around
    /// each call.
    abi: LibraryAbi,
    /// FP8 sites' gather pool (empty in the direct mode).
    pool_k: DeviceMem,
    pool_v: DeviceMem,
    /// BF16 cache read in place: pages are the cache's own 64-row runs, `slot_pages` per slot
    /// (`kv heads * rows per slot / 64`, head stride `rows per slot` rows), `total_pages` in all.
    direct: Option<Direct>,
    /// The source cache every routed site reads: requests outside it are refused.
    cache: Cache,
    /// Device table: cu_q [MAX_REQUESTS + 1] | kv_len [MAX_REQUESTS] | gather [POOL_PAGES][3] |
    /// page table [MAX_REQUESTS][width].
    table: DeviceMem,
    /// Split-KV partials: f32 O then LSE per (split, row, head).
    workspace: DeviceMem,
    staging: Vec<Staging>,
    staging_next: usize,
    launch: Option<Launch>,
    max_kv_heads: u32,
    routed: u64,
    fallbacks: u64,
    stats_at: std::time::Instant,
}

/// Slots and rows per slot of the KV caches the routed sites read (the smallest over sites).
#[derive(Clone, Copy)]
pub(super) struct Cache {
    pub(super) slots: u32,
    pub(super) rows: u32,
}

impl Cache {
    /// Both paths read the source cache at (slot, rows [0, kv_len)): a request outside it is a
    /// scheduler fault, refused before anything is enqueued.
    fn check(&self, slot: u32, kv_len: u32) -> Result<()> {
        if slot >= self.slots || kv_len > self.rows {
            return Err(RuntimeError::Rejected(format!(
                "library attention: request slot {slot} kv_len {kv_len} outside the cache \
                 ({} slots x {} rows)",
                self.slots, self.rows
            )));
        }
        Ok(())
    }

    fn check_all(&self, requests: &[attention_gemm::Request]) -> Result<()> {
        requests.iter().try_for_each(|r| self.check(r[2], r[3]))
    }
}

#[derive(Clone, Copy)]
pub(super) struct Direct {
    pub(super) kv_heads: u32,
    pub(super) kv_stride: u32,
    pub(super) slots: u32,
}

impl Direct {
    /// Pages of one slot and in the cache, and the KV head stride (elements).
    fn pages(&self, abi: &LibraryAbi) -> (u32, u32, i64) {
        let slot = match abi.layout {
            LibraryKvLayout::HeadMajor => self.kv_heads * self.kv_stride / PAGE,
            LibraryKvLayout::RowInterleaved => self.kv_stride / PAGE,
        };
        let head_stride = match abi.layout {
            LibraryKvLayout::HeadMajor => i64::from(self.kv_stride) * i64::from(abi.head_dim),
            LibraryKvLayout::RowInterleaved => i64::from(abi.head_dim),
        };
        (slot, slot * self.slots, head_stride)
    }
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

/// The declared object's bytes against the declaration: hash, ABI, container, entry roles and the
/// adapter ABI. Host only.
fn check_object(
    bytes: Vec<u8>,
    object: &SegmentObject,
) -> std::result::Result<(Image, LibraryAbi), String> {
    if object.sha256.as_deref() != Some(plow_asset::decode_objects::image_sha256(&bytes).as_str()) {
        return Err("library attention object hash mismatch".into());
    }
    let declared = LibraryAbi::parse(&object.abi)
        .ok_or_else(|| format!("library attention ABI {} unsupported", object.abi))?;
    let image = Image::parse(bytes).map_err(|e| e.to_string())?;
    if image.manifest.abi != object.abi {
        return Err(format!(
            "object states {}; the packet declares {}",
            image.manifest.abi, object.abi
        ));
    }
    let adapters = image.entry(ADAPTERS).map_err(|e| e.to_string())?;
    if adapters.role != "adapters" {
        return Err(format!("{ADAPTERS} role {}", adapters.role));
    }
    check_adapter_abi(image.blob(adapters).map_err(|e| e.to_string())?)?;
    Ok((image, declared))
}

/// Reads and checks the declared object and loads its FA4 entries (every kernel entry hash-,
/// layout-, launch-config- and exemplar-checked).
fn open(
    be: &Arc<CudaBackend>,
    assets_dir: &Path,
    object: &SegmentObject,
) -> Result<(std::path::PathBuf, Image, LibraryAbi, Fa4)> {
    let library = assets_dir.join(&object.file);
    let rejected = |why: String| RuntimeError::Rejected(format!("{}: {why}", library.display()));
    let bytes = std::fs::read(&library).map_err(|e| rejected(e.to_string()))?;
    let (image, declared) = check_object(bytes, object).map_err(rejected)?;
    be.bind()?;
    let fa4 = Fa4::load(
        be,
        &image,
        Geometry {
            head_dim: declared.head_dim,
            heads: declared.heads,
            f16: declared.f16,
        },
    )?;
    Ok((library, image, declared, fa4))
}

impl LibAttention {
    /// Checks the declared object as `load` does when no site routes to it (`PLOW_PF_ATTN_LIBRARY=0`):
    /// a packet never carries an object the runtime would refuse.
    pub(super) fn validate(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        object: &SegmentObject,
    ) -> Result<()> {
        let (library, _, _, mut fa4) = open(be, assets_dir, object)?;
        fa4.unload(be)?;
        tracing::info!(
            library = %library.display(),
            "library prefill attention object validated; no site routes to it"
        );
        Ok(())
    }

    /// The packet's declared library object (`assets_dir`-relative), hash- and ABI-checked.
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        object: &SegmentObject,
        max_kv_heads: u32,
        direct: Option<Direct>,
        cache: Cache,
    ) -> Result<Self> {
        let (library, image, declared, fa4) = open(be, assets_dir, object)?;
        let adapters = {
            let blob = image.blob(image.entry(ADAPTERS)?)?;
            let module = be.module_load(blob)?;
            let f = |name: &str| be.get_function(&module, name);
            Adapters {
                gather: f("plow_kv_dequant_pages")?,
                zero_tail: f("plow_kv_zero_tail")?,
                convert: f("plow_cvt_rows")?,
                module: Some(module),
            }
        };
        let f16 = declared.f16;
        if direct.is_none() && !gathers(&declared) {
            return Err(RuntimeError::Rejected(format!(
                "{}: an FP8 KV cache needs the hd512 head-major gather",
                library.display()
            )));
        }
        if cache.slots == 0 || cache.rows == 0 || cache.rows % PAGE != 0 {
            return Err(RuntimeError::Rejected(
                "library attention: source cache geometry".into(),
            ));
        }
        if let Some(d) = direct {
            if (d.slots, d.kv_stride) != (cache.slots, cache.rows) {
                return Err(RuntimeError::Rejected(
                    "library attention: direct geometry differs from the source cache".into(),
                ));
            }
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
            u64::from(max_kv_heads) * u64::from(POOL_ROWS) * u64::from(declared.head_dim) * 2
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
        LOADED.store(true, Ordering::Relaxed);
        tracing::info!(
            library = %library.display(),
            pool_mib = (2 * pool_bytes) >> 20,
            pool = if direct.is_some() { "none (BF16 cache in place)" } else if f16 { "fp16" } else { "bf16" },
            abi = %object.abi,
            "library prefill attention (FA4) loaded"
        );
        Ok(Self {
            be: Arc::clone(be),
            fa4,
            adapters,
            abi: declared,
            pool_k,
            pool_v,
            direct,
            cache,
            table,
            workspace,
            staging,
            staging_next: 0,
            launch: None,
            max_kv_heads,
            routed: 0,
            fallbacks: 0,
            stats_at: std::time::Instant::now(),
        })
    }

    /// (launches the library served, launches that fell back to the native role).
    pub(super) fn launch_counts(&self) -> (u64, u64) {
        (self.routed, self.fallbacks)
    }

    fn log_counts(&self) {
        tracing::info!(
            served = self.routed,
            fallback = self.fallbacks,
            "library prefill attention launches"
        );
    }

    pub(super) fn unload(mut self) -> Result<()> {
        self.log_counts();
        self.fa4.unload(&self.be)?;
        match self.adapters.module.take() {
            Some(m) => self.be.module_unload(&m),
            None => Ok(()),
        }
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
        if self.stats_at.elapsed() >= STATS_EVERY {
            self.stats_at = std::time::Instant::now();
            self.log_counts();
        }
        if routed {
            self.routed += 1;
            SERVED.fetch_add(1, Ordering::Relaxed);
        } else {
            self.fallbacks += 1;
            FALLBACK.fetch_add(1, Ordering::Relaxed);
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
        // Before any fallback decision: a request outside the cache is an error, never a
        // silent native run.
        self.cache.check_all(requests)?;
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
        // The TMA paged path (KV tile = one 64-row page) indexes the table unguarded: an empty
        // split's producer still loads V at its first block, split_idx x per-split blocks, which
        // can reach n + splits - 2 for a request of n pages (the non-TMA path checks rows against
        // kv_len). Every row carries MAX_SPLITS columns past the widest request, filled with the
        // request's own last page.
        let width = width + MAX_SPLITS;
        // `>=`: a full gather list would reach PT_OFF and overwrite the word before row 0.
        if (self.direct.is_none() && pages >= POOL_PAGES)
            || u64::from(width) * requests.len() as u64
                > u64::from(POOL_PAGES) * MAX_REQUESTS as u64
        {
            return Ok(false);
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
        let mut lead = 0i32;
        for (r, &[_, _, slot_id, kvlen]) in requests.iter().enumerate() {
            let row = &mut table[r * width as usize..(r + 1) * width as usize];
            let n = kvlen.div_ceil(PAGE) as usize;
            if let Some(d) = self.direct {
                // The slot's own pages; the gather list carries (slot, kvlen) for the tail zeroing.
                let first = slot_id * d.pages(&self.abi).0;
                for (j, e) in row[..n].iter_mut().enumerate() {
                    *e = (first + j as u32) as i32;
                }
                gather.extend([slot_id as i32, kvlen as i32, 0]);
            } else {
                for (j, e) in row[..n].iter_mut().enumerate() {
                    *e = (gather.len() / 3) as i32;
                    let j = j as u32;
                    gather.extend([
                        slot_id as i32,
                        (j * PAGE) as i32,
                        (kvlen - j * PAGE).min(PAGE) as i32,
                    ]);
                }
            }
            // Entries past the request's pages repeat its last page: a page id the kernel reads
            // there (masked by kv_len) stays inside this request's mapped rows.
            let last = row[n - 1];
            row[n..].fill(last);
            // n_block_max == 0 makes the first load read entry -1 (the word before row 0).
            if r == 0 {
                lead = row[0];
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
        host[PT_OFF as usize - 4..PT_OFF as usize].copy_from_slice(&lead.to_le_bytes());
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
            (PT_OFF - 4, PT_OFF + pt_bytes),
        ];
        self.be
            .poison_table(self.table.base, self.table.len, stream)?;
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
        if site.kv_stride < self.cache.rows {
            return Err(RuntimeError::Rejected(
                "library attention: site cache shorter than the checked bound".into(),
            ));
        }
        if site.kv_heads > self.max_kv_heads {
            return Err(RuntimeError::Rejected(
                "library attention: KV heads exceed the pool".into(),
            ));
        }
        let (k_pages, v_pages, num_pages, head_stride) = match (self.direct, site.scales) {
            (Some(d), None) => {
                // Rows past each kvlen up to its page end: zero, since the head-major entries
                // load whole pages and 0 x NaN would poison P.V (ownership: module doc).
                let args = ZeroTailArgs {
                    k: site.k,
                    v: site.v,
                    list: self.table.base + GATHER_OFF,
                    n: l.requests,
                    kv_heads: site.kv_heads,
                    kv_stride: site.kv_stride,
                    _pad: 0,
                };
                if self.abi.layout == LibraryKvLayout::HeadMajor && l.requests > 0 {
                    let mut args = args;
                    let mut params = [&mut args as *mut ZeroTailArgs as *mut c_void];
                    self.be.launch_kernel_grid(
                        self.adapters.zero_tail,
                        [l.requests, site.kv_heads, 2],
                        ADAPTER_THREADS,
                        0,
                        &mut params,
                        Some(stream),
                    )?;
                }
                let (_, total, head_stride) = d.pages(&self.abi);
                (site.k, site.v, total, head_stride)
            }
            (None, Some(scales)) => {
                self.gather(site, scales, &l, stream)?;
                (
                    self.pool_k.base,
                    self.pool_v.base,
                    POOL_PAGES,
                    i64::from(POOL_PAGES) * i64::from(PAGE * self.abi.head_dim),
                )
            }
            _ => {
                return Err(RuntimeError::Rejected(
                    "library attention: site KV type differs from the engine's".into(),
                ))
            }
        };
        let row_bytes = u64::from(site.heads * self.abi.head_dim) * 2;
        let (t0, t1) = l.tail;
        if t1 > t0 {
            self.be.memset_d8_async(
                site.output + t0 * row_bytes,
                0,
                ((t1 - t0) * row_bytes) as usize,
                stream,
            )?;
        }
        if site.heads != self.abi.heads
            || (self.abi.layout == LibraryKvLayout::RowInterleaved
                && head_stride != i64::from(self.abi.head_dim))
        {
            return Err(RuntimeError::Rejected(
                "library attention: site geometry differs from the object's".into(),
            ));
        }
        let splits = l.splits(&self.abi, site.kv_heads, self.workspace.len);
        let rows_q = (site.q + l.head * row_bytes, (t0 - l.head) * row_bytes);
        let rows_o = (site.output + l.head * row_bytes, (t0 - l.head) * row_bytes);
        if self.abi.f16 {
            self.convert(rows_q, true, stream)?;
        }
        let opart = self.workspace.base;
        let lse = opart
            + u64::from(splits) * u64::from(l.rows) * u64::from(site.heads * self.abi.head_dim) * 4;
        let call = Call {
            q: site.q,
            k: k_pages,
            v: v_pages,
            o: if splits > 1 { opart } else { site.output },
            lse,
            out: site.output,
            cu_q: self.table.base + CU_OFF,
            used_k: self.table.base + LEN_OFF,
            page_table: self.table.base + PT_OFF,
            total_q: l.rows,
            requests: l.requests,
            width: l.width,
            pages: num_pages,
            head_stride: head_stride as u64,
            splits,
            scale: site.scale,
        };
        if let Err(e) = self.fa4.run(&self.be, site.kv_heads, &call, stream) {
            if self.abi.f16 {
                let _ = self.convert(rows_q, false, stream);
            }
            return Err(e);
        }
        if self.abi.f16 {
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
            f16: u32::from(self.abi.f16),
        };
        if l.pages == 0 || site.kv_heads == 0 {
            return Ok(());
        }
        let mut args = args;
        let mut params = [&mut args as *mut DequantArgs as *mut c_void];
        self.be.launch_kernel_grid(
            self.adapters.gather,
            [l.pages, site.kv_heads, 2],
            ADAPTER_THREADS,
            0,
            &mut params,
            Some(stream),
        )
    }

    /// In-place bf16 <-> fp16 of `(address, bytes)` (16 B multiple).
    fn convert(
        &self,
        (address, bytes): (u64, u64),
        to_f16: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        let mut n8 = bytes / 16;
        if n8 == 0 {
            return Ok(());
        }
        let mut p = address;
        let mut to = u32::from(to_f16);
        let grid = n8.div_ceil(u64::from(ADAPTER_THREADS)).min(1056) as u32;
        let mut params = [
            &mut p as *mut u64 as *mut c_void,
            &mut n8 as *mut u64 as *mut c_void,
            &mut to as *mut u32 as *mut c_void,
        ];
        self.be.launch_kernel(
            self.adapters.convert,
            grid,
            ADAPTER_THREADS,
            0,
            &mut params,
            Some(stream),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_bounds_refuse_slots_and_rows_outside() {
        let cache = Cache {
            slots: 4,
            rows: 8192,
        };
        assert!(cache.check(3, 8192).is_ok());
        assert!(cache.check(0, 0).is_ok());
        assert!(cache.check(4, 1).is_err());
        assert!(cache.check(0, 8193).is_err());
        assert!(cache.check(u32::MAX, 1).is_err());
        // A request the planner would fall back on does not hide a later one outside the cache.
        assert!(cache.check_all(&[[0, 0, 0, 0], [0, 1, 4, 1]]).is_err());
        assert!(cache.check_all(&[[0, 1, 0, 1], [1, 1, 3, 8192]]).is_ok());
    }

    /// A PLOWAOT1 container with `kv_adapters` (role `role`) holding `blob`.
    fn container(abi: &str, role: &str, blob: &[u8]) -> Vec<u8> {
        let manifest = serde_json::json!({
            "format": 1, "host_stub": "fa4_sm90_fwd", "host_stub_version": 1,
            "upstream": {}, "abi": abi,
            "entries": [{
                "name": "kv_adapters", "role": role, "kv_heads": 0, "symbol": "", "params": [],
                "sha256": plow_asset::decode_objects::image_sha256(blob),
                "offset": 0, "bytes": blob.len(), "exemplars": [],
            }],
        })
        .to_string();
        let mut out = b"PLOWAOT1".to_vec();
        out.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
        out.extend_from_slice(manifest.as_bytes());
        out.extend_from_slice(blob);
        out
    }

    #[test]
    fn declared_object_is_checked_against_the_declaration() {
        const ABI: &str =
            "lib_flash_prefill_v2:fa4:hd=512:heads=16:kv_heads=1,2:layout=head_major:dtype=bf16";
        let declare = |bytes: &[u8], abi: &str| SegmentObject {
            abi: abi.into(),
            file: "plow_fa4_hd512.aot".into(),
            sha256: Some(plow_asset::decode_objects::image_sha256(bytes)),
            promote_k512: None,
            attention: None,
            gemm: None,
            decode_plan: None,
        };
        let check = |bytes: Vec<u8>, object: &SegmentObject| match check_object(bytes, object) {
            Ok(_) => String::new(),
            Err(why) => why,
        };
        let bytes = container(ABI, "adapters", b"not a cubin");
        // Reaches the adapter ABI check, which a non-cubin blob fails.
        assert!(check(bytes.clone(), &declare(&bytes, ABI)).contains("plow_kv_dequant_abi"));
        let mut object = declare(&bytes, ABI);
        object.sha256 = None;
        assert!(check(bytes.clone(), &object).contains("hash"));
        let mut tampered = bytes.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(check(tampered, &declare(&bytes, ABI)).contains("hash"));
        let other = ABI.replace("heads=16", "heads=8");
        let restated = container(&other, "adapters", b"not a cubin");
        assert!(check(restated.clone(), &declare(&restated, ABI)).contains("declares"));
        assert!(check(bytes.clone(), &declare(&bytes, "fa4:v1")).contains("unsupported"));
        let renamed = container(ABI, "whole", b"not a cubin");
        assert!(check(renamed.clone(), &declare(&renamed, ABI)).contains("role"));
    }

    #[test]
    fn adapter_abi_needs_the_stated_version() {
        assert!(check_adapter_abi(b"not a cubin").is_err());
        assert!(check_adapter_abi(&[]).is_err());
    }
}
