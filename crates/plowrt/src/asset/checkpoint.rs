//! Safetensors checkpoint loading: mmap every `*.safetensors` shard once,
//! parse header metadata up front, and serve tensor bytes as zero-copy slices.

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use crate::{Result, RuntimeError};

/// Accumulated per-worker `Checkpoint::populate` time + bytes.
///
/// `ns` is the **sum of Instant durations across all prefetch threads** — a
/// parallel worker-time total that can far exceed wall clock (N workers busy
/// concurrently). It is **not** Disk→RAM latency.
///
/// Wall-clock Disk→RAM (start → Prefetcher Drop/join) is measured in
/// `GpuEngine::load` and is what to use for benchmarking model-load latency.
pub(crate) struct PrefetchStats {
    pub ns: AtomicU64,
    pub bytes: AtomicU64,
}

impl PrefetchStats {
    pub fn new() -> Self {
        Self {
            ns: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}

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
    /// Payload dtype is OCP e4m3 (`F8_E4M3`). Kept so the loader can scrub 0x80
    /// (`-0`) bytes at stage time — see `is_fp8_e4m3` and
    /// `HsaUploadRing::push_scrub_fp8_neg0`.
    fp8: bool,
    dtype: safetensors::Dtype,
}

/// Sub-timings for [`Checkpoint::open`] (`PLOW_LOAD_PROFILE`).
#[derive(Default, Clone, Debug)]
pub(crate) struct CheckpointOpenTiming {
    pub scan_ms: f64,
    pub mmap_ms: f64,
    pub meta_ms: f64,
    pub index_ms: f64,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        Self::open_with_timing(dir, None)
    }

    /// Same as [`Self::open`], optionally filling per-phase Instant breakdowns.
    pub fn open_with_timing(
        dir: &Path,
        mut timing: Option<&mut CheckpointOpenTiming>,
    ) -> Result<Checkpoint> {
        let t0 = std::time::Instant::now();
        tracing::info!(dir = %dir.display(), "opening safetensors checkpoint...");
        let mut shards = Vec::new();
        let mut index = FxHashMap::default();

        let t_scan = std::time::Instant::now();
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|source| RuntimeError::Io {
                path: dir.to_path_buf(),
                source,
            })?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        paths.sort();
        if let Some(t) = timing.as_mut() {
            t.scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;
        }
        if paths.is_empty() {
            return Err(RuntimeError::Device(format!(
                "no *.safetensors in {}",
                dir.display()
            )));
        }
        tracing::info!(shards = paths.len(), "found safetensors shards");
        for path in &paths {
            let t_mmap = std::time::Instant::now();
            let file = std::fs::File::open(path).map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
            // SAFETY: read-only checkpoint mapping held for the engine lifetime.
            let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
            if let Some(t) = timing.as_mut() {
                t.mmap_ms += t_mmap.elapsed().as_secs_f64() * 1e3;
            }

            let t_meta = std::time::Instant::now();
            let (header_len, meta) =
                safetensors::SafeTensors::read_metadata(&map).map_err(|e| {
                    RuntimeError::Device(format!("safetensors {}: {e}", path.display()))
                })?;
            if let Some(t) = timing.as_mut() {
                t.meta_ms += t_meta.elapsed().as_secs_f64() * 1e3;
            }

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
            let t_idx = std::time::Instant::now();
            index.reserve(tensors.len());
            for (name, info) in tensors {
                index.insert(
                    name.clone(),
                    Entry {
                        shard,
                        range: info.data_offsets.0..info.data_offsets.1,
                        shape: info.shape.clone(),
                        fp8: matches!(info.dtype, safetensors::Dtype::F8_E4M3),
                        dtype: info.dtype,
                    },
                );
            }
            if let Some(t) = timing.as_mut() {
                t.index_ms += t_idx.elapsed().as_secs_f64() * 1e3;
            }
            shards.push((map, data_off));
        }
        tracing::info!(
            shards = shards.len(),
            tensors = index.len(),
            ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3).as_str(),
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

    pub fn dtype(&self, name: &str) -> Option<safetensors::Dtype> {
        self.index.get(name).map(|e| e.dtype)
    }

    /// Is `name` an OCP e4m3 (`F8_E4M3`) payload? Drives the loader's 0x80
    /// (`-0`) scrub — value-identical, and what lets the CDNA3 grouped-GEMM
    /// staging decode drop its neg-0 mask (runtime/amd/op_moe.h).
    pub fn is_fp8_e4m3(&self, name: &str) -> bool {
        self.index.get(name).is_some_and(|e| e.fp8)
    }

    /// Tensor bytes **and** shape — what a row-parallel shard needs.
    pub fn tensor_ex(&self, name: &str) -> Option<(&[u8], &[usize])> {
        let e = self.index.get(name)?;
        let (map, off) = &self.shards[e.shard];
        let bytes = map.get(off + e.range.start..off + e.range.end)?;
        Some((bytes, &e.shape))
    }

    /// Locate a sub-range of a tensor as a [`Span`] the prefetcher can carry to
    /// another thread.
    ///
    /// `off`/`len` are a sub-range **of the tensor**, because a column-parallel
    /// rank binds a contiguous 1/tp slice and prefetching the whole tensor would
    /// read four times what that rank will touch.
    pub fn span(&self, name: &str, off: usize, len: usize) -> Option<Span> {
        let e = self.index.get(name)?;
        let (map, base) = &self.shards[e.shard];
        let lo = base + e.range.start + off;
        let hi = lo.saturating_add(len).min(base + e.range.end);
        if hi <= lo || hi > map.len() {
            return None;
        }
        Some(Span {
            shard: e.shard,
            off: lo,
            len: hi - lo,
        })
    }

    /// Fault `span` in — page cache AND this process's page tables — and block
    /// until it is resident. Returns `false` if the kernel refused the hint.
    ///
    /// # Why `MADV_POPULATE_READ` and not `MADV_WILLNEED`
    ///
    /// Both were measured on this box, cold, on untouched 9.5 GiB shards of
    /// GLM-5.2 (`/dev/nvme0n1`, ext4 on LVM; `mincore` confirms 0 % resident
    /// before each run):
    ///
    /// | | 1 thr | 4 | 16 | 32 |
    /// |---|---|---|---|---|
    /// | mmap fault (what the loader did) | 2.52 | 2.65 | 3.13 | 2.10 GiB/s |
    /// | `MADV_WILLNEED` then fault | 2.65 | — | — | — |
    /// | `MADV_POPULATE_READ` | 2.27 | 2.31 | **5.30** | **5.34** GiB/s |
    /// | `pread` (reference ceiling) | 1.51 | 3.78 | 6.06 | 6.13 GiB/s |
    ///
    /// Two things fall out and both shaped this code:
    ///
    /// 1. **`MADV_WILLNEED` is worth nothing here** — 2.65 vs 2.49 GiB/s, inside
    ///    the noise. It submits readahead in 2 MiB chunks from the calling
    ///    thread and the toucher catches up immediately, so the queue never gets
    ///    deep. It was implemented, measured, and removed.
    /// 2. **The drive needs ~16 concurrent readers to reach its ~6 GiB/s**, and
    ///    NOTHING single-threaded gets past 2.9. That is why this is called from
    ///    a thread pool rather than inline: the depth has to come from
    ///    concurrency, and no amount of hinting from one thread supplies it.
    ///
    /// Warm it still pays — 33.8 GiB/s vs 15.9 for a userspace fault loop — for
    /// a different reason. Warm the pages are in cache but this process's PTEs
    /// are empty, and populating 44 M of them one minor fault at a time is what
    /// cost a warm rank 44 s. One syscall per 1.4 MiB does the same work in a
    /// kernel loop with no trap per page.
    ///
    /// Not `MADV_HUGEPAGE`: this box runs 5.15 with `CONFIG_READ_ONLY_THP_FOR_FS`
    /// unset, and ext4 has no large-folio page cache before 6.x, so a huge-page
    /// hint on a file-backed mapping does nothing at all here. It would also aim
    /// at the wrong term — the cold cost is device I/O, which a huge page does
    /// not reduce by one byte.
    pub fn populate(&self, span: Span) -> bool {
        let (map, _) = &self.shards[span.shard];
        map.advise_range(memmap2::Advice::PopulateRead, span.off, span.len)
            .is_ok()
    }
}

/// A byte range of one mmap'd shard, resolved away from the name so it can be
/// handed to a prefetch thread without borrowing the [`Checkpoint`] index.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    shard: usize,
    off: usize,
    len: usize,
}

/// A pool of threads that fault checkpoint ranges in ahead of the loader.
///
/// The loader pushes the span it will read `depth` tensors from now and carries
/// on copying; by the time it gets there the pages are resident and its own
/// access is a hit. See [`Checkpoint::populate`] for the measurements that say
/// this has to be a POOL and not an inline hint.
///
/// Bounded queue, deliberately: an unbounded one would let the prefetcher run
/// arbitrarily far ahead of a slow upload and evict the very pages it read. The
/// bound is the whole back-pressure mechanism.
pub(crate) struct Prefetcher {
    tx: Option<std::sync::mpsc::SyncSender<Span>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// How many tensors ahead of the copy the prefetch pool runs. `0` disables it.
///
/// Sized in TENSORS because that is the unit the loops walk; an expert
/// projection here is ~1.4 MiB, so the default leaves a few hundred MiB of reads
/// outstanding — past the knee of the scaling table in
/// [`Checkpoint::populate`], and small enough that the lookahead cannot evict
/// what it just read.
pub(crate) fn prefetch_depth() -> usize {
    crate::config::RuntimeConfig::get().prefetch
}

/// Prefetch threads per rank. The measured knee is ~16; below it the drive runs
/// at latency rather than bandwidth.
pub(crate) fn prefetch_threads() -> usize {
    crate::config::RuntimeConfig::get().prefetch_threads
}

/// Carve the weights out of ONE device allocation instead of asking the driver
/// per tensor. `--rt-weight-slab false` / `PLOW_WEIGHT_SLAB=0` turns it off on
/// both backends.
///
/// On by default because the per-tensor path wastes memory to allocator
/// rounding — modestly on CUDA (322 MiB → 21 MiB on a 12B model), badly on ROCr,
/// which rounds every request under 2 MiB up to the next POWER OF TWO (1325 MiB
/// per card on Kimi-K3).
///
/// Whether it also saves TIME differs by backend, and neither answer generalises:
/// on CUDA it does not (the driver charges by committed bytes, 1.97 s → 1.92 s),
/// on ROCr it does, by ~7–8.5 s per rank. See `exec::gpu`'s and `exec::amd`'s
/// `_weight_slab` for the measurements and for why the AMD one cannot be
/// reproduced by a standalone allocation benchmark.
///
/// The off switch exists because one allocation is a strictly stronger demand on
/// the card than many small ones: a fragmented or shared GPU can satisfy the
/// second and refuse the first. The loader already falls back on its own when
/// the allocation is refused outright, so this is for the case where it
/// SUCCEEDS and is still the wrong thing — a co-tenant that needed the hole.
pub(crate) fn weight_slab_enabled() -> bool {
    crate::config::RuntimeConfig::get().weight_slab
}

/// `--nv-weight-vmm false` / `PLOW_WEIGHT_VMM=0` drops the CUDA weight slab
/// from VMM lazy commit back to one flat `cuMemAlloc`. Default ON: the VMM
/// slab reserves VA in µs and commits pages on a background thread overlapped
/// with the upload, which is what actually removes the §4b commit stall
/// (`exec::gpu::_weight_slab`). The off switch exists for driver-regression
/// triage, not tuning.
#[cfg(feature = "cuda")]
pub(crate) fn weight_vmm_enabled() -> bool {
    crate::config::RuntimeConfig::get().nv.weight_vmm
}

/// `--nv-upload-direct false` / `PLOW_UPLOAD_DIRECT=0` forces the
/// pinned-staging upload even on a device
/// whose DMA engines read pageable memory coherently
/// (`Backend::coherent_host_dma` — GH200 ATS). Default on: the direct copy
/// takes its source straight from the checkpoint mmap at link speed
/// (332 GiB/s measured vs 13 GiB/s staged). The off switch exists for
/// driver-regression triage, not tuning.
#[cfg(feature = "cuda")]
pub(crate) fn upload_direct_enabled() -> bool {
    crate::config::RuntimeConfig::get().nv.upload_direct
}

/// The AMD loader's gate for the same slab, with the opposite default:
/// **opt-in** (`--amd-weight-vmm` / `PLOW_WEIGHT_VMM=1`). ROCr's
/// `hsa_amd_vmem_*` surface is
/// resolved and drives VmmKv already, but the lazy-commit weight slab has not
/// been measured on AMD hardware — and §4b's AMD numbers say the flat slab
/// already saved 7–8.5 s/rank there, so the residual win is unknown. Flip the
/// default only with a measurement on a gfx950 box.
#[cfg(feature = "hsa")]
pub(crate) fn weight_vmm_amd_enabled() -> bool {
    crate::config::RuntimeConfig::get().amd.weight_vmm
}

impl Prefetcher {
    /// `threads = 0` disables prefetching and returns `None`, which every call
    /// site already handles — the loader then faults pages in itself, correctly
    /// and slowly, exactly as it did before.
    ///
    /// `stats`: when `Some`, workers accumulate populate Instant durations into
    /// `PrefetchStats::ns` (parallel sum across threads — not wall latency).
    /// Wall-clock Disk→RAM is timed in `GpuEngine::load` around start→Drop/join.
    /// Pass `None` when stats are not needed.
    pub fn start(
        ckpt: std::sync::Arc<Checkpoint>,
        threads: usize,
        queue: usize,
        stats: Option<std::sync::Arc<PrefetchStats>>,
    ) -> Option<Prefetcher> {
        if threads == 0 {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<Span>(queue.max(threads));
        // `Receiver` is single-consumer; a mutex around it is what makes the
        // fan-out possible. Contention is a non-issue — one lock per 1.4 MiB of
        // I/O, against a hold time of a `VecDeque` pop.
        let rx = std::sync::Arc::new(parking_lot::Mutex::new(rx));
        let workers = (0..threads)
            .map(|_| {
                let (rx, ckpt, stats) = (
                    std::sync::Arc::clone(&rx),
                    std::sync::Arc::clone(&ckpt),
                    stats.clone(),
                );
                std::thread::spawn(move || {
                    loop {
                        let job = { rx.lock().recv() };
                        let Ok(span) = job else { return };
                        // A refused hint (no `MADV_POPULATE_READ` before 5.14)
                        // is not an error: the loader's own fault still gets the
                        // right bytes. Keep draining so the sender never blocks.
                        if let Some(s) = &stats {
                            // Per-call Instant: summed across workers → parallel
                            // worker time, not wall-clock Disk→RAM latency.
                            let t = std::time::Instant::now();
                            ckpt.populate(span);
                            s.ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            s.bytes.fetch_add(span.len as u64, Ordering::Relaxed);
                        } else {
                            ckpt.populate(span);
                        }
                    }
                })
            })
            .collect();
        Some(Prefetcher {
            tx: Some(tx),
            workers,
        })
    }

    /// Queue a span, blocking while the pool is `queue` jobs behind.
    pub fn push(&self, span: Span) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(span);
        }
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Dropping the sender is what ends the workers' `recv`. Joining them
        // before the `Arc<Checkpoint>` can fall is not optional: a worker is
        // inside `madvise` on a mapping the checkpoint owns.
        self.tx = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
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
        let Ok(bytes) = std::fs::read(dir.join(file)) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    return vec![id as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> = a
                    .iter()
                    .filter_map(|x| x.as_u64().map(|v| v as u32))
                    .collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}

/// Stop ids a served CHAT turn needs beyond `eos_token_id`.
///
/// `generation_config.json` names the id that ends a *sequence*. Some chat formats close a
/// structured turn before they get there, and a served completion that runs to the sequence eos
/// leaks the framing into the user's text.
///
/// Kimi-K3 is the live case and the only one so far. Its turn is XTML-ish
/// (`encoding_k3.py::build_chat_segments`): the assistant opens `<|open|>response<|sep|>` and the
/// answer ends at the MATCHING `<|close|>` — after which the template goes on to emit
/// `response<|sep|><|close|>message<|sep|>` and only then `<|end_of_msg|>` (163586), which is what
/// `eos_token_id` names. Stopping only at eos returned
///
/// ```text
/// The capital of France is Paris.<|close|>response<|sep|><|close|>message<|sep|>
/// ```
///
/// — a correct answer with four markers of channel bookkeeping stapled to it.
/// (Fenced `text`, not indented: an indented block is a RUST block to rustdoc,
/// which then tries to compile this sentence and fails the doctest run.)
///
/// Keyed on the CHECKPOINT's own tokens rather than on a model name: the extra stop is added only
/// when this checkpoint both declares `<|end_of_msg|>` as its eos AND ships a `<|close|>` token,
/// which is exactly the K3 turn structure and cannot fire on a checkpoint shaped differently.
/// `serve::chat::k3_chat_prompt` renders the matching prompt; the two must stay in step, so if
/// that template ever opens the `think` channel instead, this rule needs revisiting — `<|close|>`
/// would then end the THOUGHT, not the answer.
pub(crate) fn chat_stop_ids(dir: &Path, eos: &[u32]) -> Vec<u32> {
    let Ok(bytes) = std::fs::read(dir.join("tokenizer_config.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let Some(added) = v.get("added_tokens_decoder").and_then(|a| a.as_object()) else {
        return Vec::new();
    };
    let id_of = |want: &str| -> Option<u32> {
        added.iter().find_map(|(k, e)| {
            (e.get("content").and_then(|c| c.as_str()) == Some(want))
                .then(|| k.parse::<u32>().ok())
                .flatten()
        })
    };
    match (id_of("<|end_of_msg|>"), id_of("<|close|>")) {
        (Some(eom), Some(close)) if eos.contains(&eom) => vec![close],
        _ => Vec::new(),
    }
}
